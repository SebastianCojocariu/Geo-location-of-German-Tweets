import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Embedding, Concatenate, Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, GaussianNoise, SpatialDropout1D

class CharCNNTrainer(object):
    def __init__(self, alphabet_characters_length, alphabet_words_length, config, no_classes=None):

        # the length of the alphabets (charAlphabet and wordAlphabet)
        self.alphabet_characters_length = alphabet_characters_length
        self.alphabet_words_length = alphabet_words_length

        # total numbers of characters/words per tweet (PADDING or TRUNCATION might be applied)
        self.total_characters = config["total_characters"]
        self.total_words = config["total_words"]

        # the type of the task ("regression" or "classification" using the regions from KMeans as classes)
        self.task_type = config["task_type"]

        # regularization techniques
        self.initial_lr = config["initial_lr"]
        self.weight_decay = config["weight_decay"]
        self.dropout = config["dropout"]
        self.gaussian_noise = config["gaussian_noise"]
        self.spatial_dropout = config["spatial_dropout"]

        # loss function
        self.loss = config["loss"]

        # 2 submodels at different levels: character level and word level
        # Is used to enable training on each of them or only one of them
        self.character_level_CNN = config["character_level_CNN"]
        self.word_level_CNN = config["word_level_CNN"]

        # assert there is at least one of the submodels active
        assert (self.character_level_CNN or self.word_level_CNN)

        # number of classes. Is not relevant during regression
        self.no_classes = no_classes

        # the embedding size of each dictionary
        self.embedding_characters_dimension = 128
        self.embedding_words_dimension = 128

        # the convolutional layers at character level
        self.convolutional_layers_characters = [
            [[1024, 9, "relu"], 3],
            [[768, 7, "relu"], 3],
            [[512, 7, "relu"], 3]
        ]

        # the convolutional layers at word level
        self.convolutional_layers_words = [
            [[1024, 7, "relu"], 3],
            [[768, 5, "relu"], 2],
            [[512, 3, "relu"], -1]
        ]

        # the fully connected layers at each level. Used to map the flatten of each last convolutional output.
        # from practice is better to use the same values for both of them
        # (to not give one submodule more importance)
        self.fully_connected_characters = [512, "relu"]
        self.fully_connected_words = [512, "relu"]

        # the final fully connected layers
        self.fully_connected_final = [
            [512, "relu"],
            [256, "relu"],
            [128, "relu"],
            [64, "relu"]
        ]

        self._create_model()

    def _create_model(self):

        output_characters, output_words = None, None

        ####### Character Level ######

        # Input layer
        input_characters = Input(shape=(self.total_characters,), name='input_characters', dtype='int64')

        if self.character_level_CNN:
            res_characters = Embedding(self.alphabet_characters_length + 1, self.embedding_characters_dimension,
                               input_length=self.total_characters)(input_characters)
            res_characters = Reshape((self.total_characters, self.embedding_characters_dimension))(res_characters)

            # Convolutional layers
            for cl in self.convolutional_layers_characters:
                # Convolutional layer
                res_characters = Conv1D(cl[0][0], cl[0][1], padding="same", activation=cl[0][2])(res_characters)

                # Max Pooling
                if cl[1] > 0:
                    res_characters = MaxPooling1D(cl[1], cl[1])(res_characters)

                # Apply regularization techniques: GaussianNoise and spatialDropout
                if self.gaussian_noise > 0:
                    res_characters = GaussianNoise(stddev=self.gaussian_noise)(res_characters)
                if self.spatial_dropout > 0:
                    res_characters = SpatialDropout1D(rate=self.spatial_dropout)(res_characters)

            # Flatten the features
            res_characters = Flatten()(res_characters)
            output_characters = Dense(self.fully_connected_characters[0], activation=self.fully_connected_characters[1])(res_characters)


        ####### Word Level #######

        # Input layer
        input_words = Input(shape=(self.total_words,), name='input_words', dtype='int64')

        if self.word_level_CNN:
            res_words = Embedding(input_dim=self.alphabet_words_length + 1, output_dim=self.embedding_words_dimension,
                               input_length=self.total_words)(input_words)
            res_words = Reshape((self.total_words, self.embedding_words_dimension))(res_words)


            # Convolutional blocks
            for cl in self.convolutional_layers_words:
                # Convolutional layer
                res_words = Conv1D(cl[0][0], cl[0][1], padding="same", activation=cl[0][2])(res_words)

                # Max Pooling
                if cl[1] > 0:
                    res_words = MaxPooling1D(cl[1], cl[1])(res_words)

                # Apply regularization techniques: GaussianNoise and spatialDropout
                if self.gaussian_noise > 0:
                    res_words = GaussianNoise(stddev=self.gaussian_noise)(res_words)
                if self.spatial_dropout > 0:
                    res_words = SpatialDropout1D(rate=self.spatial_dropout)(res_words)

            res_words = Flatten()(res_words)
            output_words = Dense(self.fully_connected_words[0], activation=self.fully_connected_words[1])(res_words)

        ####### Concatenate outputs #######

        outputs_list = [output for output in [output_characters, output_words] if output != None]

        res = Concatenate(axis=1)(outputs_list) if len(outputs_list) == 2 else outputs_list[0]

        # Final fully connected layers
        for [fl, activation_flag] in self.fully_connected_final:
            res = Dense(fl, activation=activation_flag)(res)

            # Apply dropout
            if self.dropout > 0:
                res = Dropout(self.dropout)(res)

        # Output layer
        if self.task_type.lower() == "regression":
            predictions = Dense(2)(res)
            self.metrics = ["mean_squared_error", 'mean_absolute_error']


        elif self.task_type.lower() == "classification":
            predictions = Dense(self.no_classes, activation="softmax")(res)
            self.metrics = ['accuracy']

        # Build and compile the models
        model = Model(inputs=[input_characters, input_words], outputs=predictions)
        optimizer = tfa.optimizers.AdamW(learning_rate=self.initial_lr, weight_decay=self.weight_decay)

        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        self.model = model
        self.model.summary()


