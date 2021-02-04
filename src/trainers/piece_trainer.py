import numpy as np

from trainers.train import Train
from utils import (
    load_data_file,
    flatten_coord_to_target_array,
    get_data_file_sample_length,
)


class PieceTrainer(Train):
    def __init__(self, model, config):
        super(PieceTrainer, self).__init__(model, config)

    def prep_train(self):
        self.model.model.compile(
            loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"]
        )

    def train(self, train_file_pairs, valid_file_pairs):
        batch_size = self.config.batch_size
        train_file_pairs = list(train_file_pairs)
        valid_file_pairs = list(valid_file_pairs)

        train_gen = self.generate_data_from_file_pair(train_file_pairs)
        valid_gen = self.generate_data_from_file_pair(valid_file_pairs)

        train_SPE = sum(get_data_file_sample_length("t", x) for x in train_file_pairs)
        valid_SPE = sum(get_data_file_sample_length("v", x) for x in valid_file_pairs)

        train_valid_SPE = [train_SPE, valid_SPE]
        for i, steps_per_epoch in enumerate(train_valid_SPE):
            if steps_per_epoch % batch_size == 0:
                steps_per_epoch /= batch_size
        else:
            train_valid_SPE[i] = steps_per_epoch / batch_size + 1

        history = self.model.model.fit(
            train_gen,
            epochs=self.config.epochs,
            steps_per_epoch=train_valid_SPE[0],
            validation_data=valid_gen,
            validation_steps=train_valid_SPE[1],
        )

        print(history.history)

    def generate_data_from_file_pair(self, file_pairs):
        file_counter, sample_counter = 0, 0
        batch_size = self.config.batch_size
        while True:
            file_pair = file_pairs[file_counter]
            x, y_flat = (load_data_file(file_pair[0]), load_data_file(file_pair[1]))
            y = flatten_coord_to_target_array(y_flat)

            # TODO: investigate having to np.array the inputs. This feels strange.
            x = np.asarray(x)
            y = np.asarray(y)

            stop_index = sample_counter + batch_size
            length = len(x)
            if stop_index > length:
                yield (x[sample_counter:], y[sample_counter:])
                if file_counter < len(file_pairs) - 1:
                    file_counter += 1
                sample_counter = 0
            else:
                yield (x[sample_counter:stop_index], y[sample_counter:stop_index])
                sample_counter += batch_size
