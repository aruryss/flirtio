from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Attention, Concatenate
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FlirtReplyGenerator:
    """
    Wrapper around the seq2seq + Attention model from seq2seq.ipynb.

    Usage:
        gen = FlirtReplyGenerator(cfg)
        reply = gen.generate("hi, how are you?", flirty=True)
    """

    def __init__(self, cfg: Config, project_root: Path = None):
        self.cfg = cfg
        self.project_root = project_root or Path(__file__).resolve().parents[1]

        # paths
        gen_dir = self.project_root / "flirt-generation"
        self.weights_path = gen_dir / Path(cfg.model_paths.generation_weights)
        self.input_tok_path = gen_dir / Path(cfg.model_paths.input_tokenizer)
        self.target_tok_path = gen_dir / Path(cfg.model_paths.target_tokenizer)
        self.data_path = gen_dir / "augmented-data" / "detection_train_augmented_with_replies.csv"

        # lazy-initialized attributes
        self.input_tokenizer = None
        self.target_tokenizer = None
        self.encoder_model = None
        self.decoder_model = None
        self.max_encoder_len = None
        self.max_decoder_len = None
        self.embedding_dim = 128
        self.latent_dim = 256

        self._load_tokenizers()
        self._prepare_lengths()
        self._build_models()

    # ---------- loading tokenizers & lengths ----------

    def _load_tokenizers(self):
        with open(self.input_tok_path, "rb") as f:
            self.input_tokenizer = pickle.load(f)
        with open(self.target_tok_path, "rb") as f:
            self.target_tokenizer = pickle.load(f)
        logger.info("Loaded input/target tokenizers.")

    def _prepare_lengths(self):
        df = pd.read_csv(self.data_path)
        input_texts = list(df["input"].values)
        output_texts = list(df["reply"].values)

        target_texts = [f"<START> {t} <END>" for t in output_texts]

        encoder_seqs = self.input_tokenizer.texts_to_sequences(input_texts)
        decoder_seqs = self.target_tokenizer.texts_to_sequences(target_texts)

        self.max_encoder_len = max(len(s) for s in encoder_seqs) if encoder_seqs else 1
        self.max_decoder_len = max(len(s) for s in decoder_seqs) if decoder_seqs else 1

        logger.info(
            f"max_encoder_len={self.max_encoder_len}, "
            f"max_decoder_len={self.max_decoder_len}"
        )

    # ---------- architecture & weights ----------

    def _build_models(self):
        """Rebuild the training architecture and load saved weights."""

        input_vocab_size = len(self.input_tokenizer.word_index) + 1
        target_vocab_size = len(self.target_tokenizer.word_index) + 1

        # ENCODER
        encoder_inputs = Input(shape=(self.max_encoder_len,))
        label_inputs = Input(shape=(1,))

        encoder_embedding = Embedding(input_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        # incorporate label
        label_dense = Dense(self.latent_dim)(label_inputs)
        state_h = Concatenate()([state_h, label_dense])
        state_h = Dense(self.latent_dim)(state_h)
        state_c = Concatenate()([state_c, label_dense])
        state_c = Dense(self.latent_dim)(state_c)

        encoder_states = [state_h, state_c]

        # DECODER
        decoder_inputs = Input(shape=(self.max_decoder_len,))
        decoder_embedding = Embedding(target_vocab_size, self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        attention = Attention()
        context_vector = attention([decoder_outputs, encoder_outputs])

        decoder_combined = Concatenate()([decoder_outputs, context_vector])
        decoder_dense = Dense(target_vocab_size, activation="softmax")
        decoder_outputs = decoder_dense(decoder_combined)

        training_model = keras.Model(
            [encoder_inputs, label_inputs, decoder_inputs],
            decoder_outputs,
            name="flirty_response_seq2seq",
        )
        training_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


        training_model.load_weights(str(self.weights_path))
        logger.info(f"Loaded generation weights from {self.weights_path}")

        # ----- inference models -----
        # Encoder model
        encoder_model = keras.Model(
            [encoder_inputs, label_inputs],
            [encoder_outputs, state_h, state_c],
            name="encoder_inference",
        )

        # Decoder inference
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_hidden_state_input = Input(shape=(self.max_encoder_len, self.latent_dim))

        dec_outputs, dec_h, dec_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )
        decoder_states = [dec_h, dec_c]

        context_vector_inf = attention([dec_outputs, decoder_hidden_state_input])
        decoder_combined_inf = Concatenate()([dec_outputs, context_vector_inf])
        dec_outputs = decoder_dense(decoder_combined_inf)

        decoder_model = keras.Model(
            [decoder_inputs, decoder_hidden_state_input] + decoder_states_inputs,
            [dec_outputs] + decoder_states,
            name="decoder_inference",
        )

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    # ---------- public API ----------

    def generate(self, text: str, flirty: bool = True) -> str:
        """
        Generate a reply for input text.
        flirty=True  -> use label 1.0
        flirty=False -> use label 0.0
        """
        label = 1.0 if flirty else 0.0

        # encode input
        seq = self.input_tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=self.max_encoder_len, padding="post")
        label_seq = np.array([[label]])

        encoder_output, h, c = self.encoder_model.predict([seq, label_seq], verbose=0)

        start_id = self.target_tokenizer.word_index.get("<START>", 1)
        end_token = "<END>"

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = start_id

        decoded_words = []

        for _ in range(self.cfg.max_gen_len):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_output, h, c], verbose=0
            )
            sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
            sampled_word = self.target_tokenizer.index_word.get(sampled_token_index, "")

            if sampled_word == end_token or len(decoded_words) > self.cfg.max_gen_len:
                break

            if sampled_word not in ["<START>", "<OOV>", end_token, ""]:
                decoded_words.append(sampled_word)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        reply = " ".join(decoded_words).strip()
        logger.info(f"Generated reply: {reply}")
        return reply