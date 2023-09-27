#importowanie wymaganych bibliotek

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import random
from keras.utils import Sequence
from statsmodels.tsa.seasonal import seasonal_decompose

#utworzenie generatora szeregów czasowych z przesuwem okna co pełne 24 godziny

class TimeSeriesGeneratorEvery24h(Sequence):
    def __init__(self, data, targets, length, batch_size):
        self.data = data
        self.targets = targets
        self.length = length
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.data) - self.length) // (self.batch_size * 24) * self.batch_size

    def __getitem__(self, idx):
        data_idx = np.arange(idx * self.batch_size * 24, (idx + 1) * self.batch_size * 24, 24)
        data = np.array([self.data[i: i + self.length] for i in data_idx])
        # targets = np.array([self.targets[i + self.length: i + self.length + 24] for i in data_idx])
        targets = np.array([self.targets[i + self.length - 24: i + self.length] for i in data_idx])
        data = np.array(data)
        targets = np.array(targets)
        return data, targets

#utworzenie generatora szeregów czasowych z przesuwem okna co godzinę

class TimeSeriesGeneratorEvery1h(Sequence):
    def __init__(self, data, targets, length, batch_size):
        self.data = data
        self.targets = targets
        self.length = length
        self.batch_size = batch_size

    def __len__(self):
        return ((len(self.data) - self.length) // (self.batch_size) * self.batch_size) - 23

    def __getitem__(self, idx):
        data_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size, 24)
        data = np.array([self.data[i: i + self.length] for i in data_idx])
        # targets = np.array([self.targets[i + self.length: i + self.length + 24] for i in data_idx])
        targets = np.array([self.targets[i + self.length - 24: i + self.length] for i in data_idx])
        data = np.array(data)
        targets = np.array(targets)
        return data, targets

#deklaracja funkcji zmiany formatu z wykorzystaniem biblioteki numpy, niektóre funkcje lepiej współpracują z tym formatem

def convert_generator_to_numpy(source):
    data, targets = [], []
    for i in range(len(source)):
        x, y = source[i]
        data.append(x)
        targets.append(y)

    data = np.vstack(data)
    targets = np.vstack(targets)
    return data, targets

#deklaracja funkcji dekompozycji danych do grafu

def plot_decomposition(data, columns, colors):
    fig, ax = plt.subplots(nrows=len(columns), ncols=4, figsize=(12, 3 * len(columns)), dpi=80,
                           gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    for i, col in enumerate(columns):
        result = seasonal_decompose(data[col], model='additive', period=8760, extrapolate_trend=0)
        c = colors[i]
        result.observed.plot(ax=ax[i][0], color=c, label=col)
        result.trend.plot(ax=ax[i][1], color=c, label=col)
        result.seasonal.plot(ax=ax[i][2], color=c, label=col)
        result.resid.plot(ax=ax[i][3], color=c, label=col)

        ax[i][0].set_title('Observed')
        ax[i][1].set_title('Trend')
        ax[i][2].set_title('Seasonality')
        ax[i][3].set_title('Residuals')
        ax[i][0].legend(loc="upper left")

    plt.tight_layout()

#deklaracja funkcji wizualizacji danych wejściowych

def show_raw_visualization(data):
    if df.shape[1] % 2 != 0:
        rows = (df.shape[1] + 1) / 2
    else:
        rows = df.shape[1] / 2
    rows = int(rows)
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 18), dpi=80)
    for i in range(len(data.columns)):
        column = data.columns[i]
        color = (random.random(), random.random(), random.random())
        ax = data[column].plot(
            ax=axes[i // 2, i % 2],
            color=color,
            title='',
            rot=25,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend().remove()
        ax.set_title(column, loc="left", fontsize=12)
    plt.tight_layout()

#deklaracja funkcji mapy korelacji

def show_heatmap(data):
    plt.figure(figsize=(11, 10))
    plt.matshow(data.corr(), fignum=2, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(data.shape[1]), data.columns, fontsize=8, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=8)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Data Correlation Heatmap", fontsize=14)

#deklaracja funkcji pozyskiwania elementów o współczynniku koleracji conajmniej 0.4

def get_high_correlation_features(data, target_column, threshold=0.4):
    corr_matrix = data.corr()
    high_correlation_features = []
    for column in corr_matrix.columns:
        if column == target_column:
            continue
        if abs(corr_matrix[target_column][column]) >= threshold:
            high_correlation_features.append(column)
    high_correlation_features.append("Power")
    return high_correlation_features

#deklaracja funkcji utworzenia losowych kolorów do grafów

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = [random.random() for i in range(3)]
        colors.append(color)
    return colors

print("Wersja Python:", platform.python_version())
print("Wersja TensorFlow:", tf.version.VERSION)

print("Ilość dostępnych GPU: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("Sprawdź stronę https://www.tensorflow.org/install/source#gpu aby sprawdzić wspierane wersje oprogramowania.")
print("Tensorflow do wersji 2.10.0 ma wsparcie dla Windows w kwestii obliczeń na GPU.")

#załadowanie pliku źródłowego

df = pd.read_csv('Dane wejściowe.csv', index_col='PeriodStart', parse_dates=True)

#przywołanie wizualizacji danych wejściowych

show_raw_visualization(df)

#dodanie dwóch kolumn
#kolumna dnia roku
#kolumna godziny dnia

df['DayOfYear'] = df.index.dayofyear
df['HourOfDay'] = df.index.hour

#przywołanie wizualizacji mapy korelacji

show_heatmap(df)

#przywołanie funkcji uzyskania najbardziej skolerowanych elementów na podstawie mapy
#elementy skorelowanie z kolumną "mocy chwilowej"

high_correlation_features = get_high_correlation_features(df, "Power")

columns = high_correlation_features
colors = generate_colors(len(high_correlation_features))
plot_decomposition(df, columns, colors)

#prezentacja wszystkich wizualizacji

plt.show()

#zmiennie wykorzystywane do szybkiej kontroli nad parametrami
#zmienna a oznacza rozmiar zestawu uczącego
#zmienna b oznacza rozmiar zestawu walidującego
#100-a-b oznacza rozmiar zestawu testowego
#n_input jest zmienną wprowadzaną do generatora gdzie odpowiada za ustalenie ilości godzin wykorzystywanych do wykonania predykcji 24 godzin
#(generator jest skonstruowany tak że po wpisaniu wartości większej niż 24, kolejne godziny ponad 24 będą z dni wcześniejszych)
#n_input informuje również strukturę sieci na temat rozmiaru wektora w warstwie wejściowej
#w przypadku ZMIAN w zestawie danych wejściowych, NALEŻY ODPOWIEDNIO ZMIENIĆ WARTOŚĆ
#LSTM_nodes jest zmienną którą można szybko kontrolować ilość węzłów w każdej warstwie sieci, dla dokładniejszej kontroli można podmienić tę zmienną konkretnymi liczbami w sieci
#epochs oznacza ilość epok przez które trwała będzie nauka
#modelname oznacza to w jaki sposób utworzona zostanie nazwa pliku
#folderpath oznacza gdzie zostanie zapisany model wewnątrz folderu środowiska wirtualnego

a = 60
b = 30
n_input = 48
n_features = df.shape[1] - 1
LSTM_nodes = 512
epochs = 25
comment = f'{a},{b},{100-a-b}BILSTM2layer'
modelname = f'{n_input}Hours{LSTM_nodes}LSTM{epochs}Epochs{comment}'
folderpath = f'Modele/{modelname}/'

#przeliczanie rozmiarów danych wejściowych na podstawie zmiennych a,b
train_size = int(len(df) * a / 100)
val_size = int(len(df) * b / 100)
test_size = len(df) - train_size - val_size

#podział danych na zestawy uczące, walidujące oraz testowe
train_data = df[:train_size]
val_data = df[train_size:train_size+val_size]
test_data = df[train_size+val_size:]

#wypisanie rozmiarów zestawów
print("Train set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))

#skalowanie zestawów wedle najmniejszej i największej wartości do przedziału między 0, a 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df)

scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)
scaled_val = scaler.transform(val_data)

#wprowadzanie zestawów danych do generatów szeregów czasowych

generator_train = TimeSeriesGeneratorEvery1h(np.delete(scaled_train, df.shape[1] - 3, 1), scaled_train[:, df.shape[1] - 3], length=n_input, batch_size=1)
generator_test = TimeSeriesGeneratorEvery24h(np.delete(scaled_test, df.shape[1] - 3, 1), scaled_test[:, df.shape[1] - 3], length=n_input, batch_size=1)
generator_val = TimeSeriesGeneratorEvery24h(np.delete(scaled_val, df.shape[1] - 3, 1), scaled_val[:, df.shape[1] - 3], length=n_input, batch_size=1)

#wydrukowanie kształtu i pojedyńczych elementów generatora danych uczących, tutaj też mozna umieścić breakpoint w czasie debugowania kodu, aby potwierdzić czy
#wejścia sieci oznaczone jako x odpowiadają wyjściom sieci y
#x powinny być danymi jakie przyszły model wykorzystałby do wykonania predykcji
#y powinny być danymi jakie przyszły model będzie starał się przewidzieć
#warto też zwrócić uwagę na poprawny przesuw okna szeregów czasowych oraz przesuw lub jego brak pomiędzy danymi, w zależności od założeń
#sprawdzenie przesuwu okna szeregów czasowych oznacza sprawdzenie czy każda następna sekwencja rozpoczyna się godzinę później czy 24 godziny później względem poprzedniej sekwencji
#TimeSeriesGeneratorEvery1h powinien mieć różnicę 1 godziny pomiędzy następującymi sekwencjami, TimeSeriesGeneratorEvery24h pełne 24 godziny
#sprawdzenie przesuwu między danymi, oznacza sprawdzenie czy do predykcji mocy z dnia x, wykorzystujemy pogodę z dnia x, czy może jest to pogoda z dnia x-1

for i in range(len(generator_train)):
    x, y = generator_train[i]
    print(x.shape, y.shape)

print(len(generator_train))

#zamiana formatu z wykorzystaniem biblioteki numpy, niektóre funkcje lepiej współpracują z tym formatem

numpytraindata, numpytraintargets = convert_generator_to_numpy(generator_train)
numpytestdata, numpytesttargets = convert_generator_to_numpy(generator_test)
numpyvaldata, numpyvaltargets = convert_generator_to_numpy(generator_val)

#ustawianie warunku wczesnego zatrzymania na podstawie poprawy lub braku poprawy pierwiastka błędu średniokwadratowego w zestawie walidującym

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', verbose=1, patience=6)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.2, patience=2, min_lr=0.001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{folderpath}{modelname}.h5', monitor='val_root_mean_squared_error',save_best_only=True, mode='min', verbose=1)
callbacks = [reduce_lr, model_checkpoint, early_stop]

#struktura modelu z wykomentowanymi modelami
#aby korzystać z konkretnego modelu, należy odkomentować go i zakomentować pozostałe

with tf.device("/gpu:0"):

# Model standardowej rekurencyjnej sieci LSTM

    # model = tf.keras.Sequential([
    #             tf.keras.layers.LSTM(units=LSTM_nodes, input_shape=(n_input, n_features), dropout=0.5, kernel_initializer='glorot_uniform', activation='tanh', return_sequences=True),
    #             tf.keras.layers.LSTM(units=int(LSTM_nodes/2), dropout=0.4, activation='tanh',kernel_initializer='glorot_uniform', return_sequences=False),
    #             tf.keras.layers.Dense(units=24, kernel_initializer='glorot_normal', activation='relu'),
    #         ])

# Model sieci LSTM z enkoderem i dekoderem

    encoder_inputs = tf.keras.Input(shape=(n_input, n_features))
    encoder_lstm = tf.keras.layers.LSTM(units=LSTM_nodes, dropout=0.2, kernel_initializer='glorot_uniform',
                                        activation='tanh', return_sequences=True)(encoder_inputs)
    dropout = tf.keras.layers.Dropout(.2)(encoder_lstm)
    encoder_lstm = tf.keras.layers.LSTM(units=int(LSTM_nodes/2), dropout=0.2, activation='tanh', kernel_initializer='glorot_uniform',
                                        return_sequences=False)(dropout)

    decoder_inputs = tf.keras.layers.RepeatVector(1)(encoder_lstm)
    decoder_lstm = tf.keras.layers.LSTM(units=int(LSTM_nodes/2), dropout=0.2, activation='tanh', kernel_initializer='glorot_uniform',
                                        return_sequences=True)(decoder_inputs)
    dropout = tf.keras.layers.Dropout(.2)(decoder_lstm)
    decoder_lstm = tf.keras.layers.LSTM(units=LSTM_nodes, dropout=0.2, kernel_initializer='glorot_uniform',
                                        activation='tanh', return_sequences=True)(dropout)
    decoder_outputs = tf.keras.layers.Dense(units=24, kernel_initializer='glorot_normal', activation='relu')(decoder_lstm)

    # Add the Permute layer to reshape the output to (1, 24, 1)
    decoder_outputs = tf.keras.layers.Permute((2, 1))(decoder_outputs)
    decoder_outputs = tf.keras.layers.Reshape((24, 1))(decoder_outputs)

    model = tf.keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

# Model dwukierunkowej rekurencyjnej sieci LSTM
#
#     model = tf.keras.Sequential([
#         tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(units=LSTM_nodes, input_shape=(n_input, n_features), dropout=0.2,
#                                  kernel_initializer='glorot_uniform', activation='tanh', return_sequences=True)),
#         tf.keras.layers.Bidirectional(
#             tf.keras.layers.LSTM(units=int(LSTM_nodes/2), dropout=0.2,
#                                  kernel_initializer='glorot_uniform', activation='tanh', return_sequences=False)),
#         tf.keras.layers.Dense(units=24, kernel_initializer='glorot_normal', activation='relu')
#     ])
#
#     model.build(input_shape=(None, n_input, n_features))

#kompilacja modelu, funkcją wykorzystywaną do sprawdzenia jakości modelu (loss) jest błąd średniokwadratowy
#oprócz tego, zapisywanie są statystyki jak pierwiastek błędu średniokwadratowego, średni błąd bezwzględny

    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error', 'RootMeanSquaredError'])

    model.summary()

#nauka modelu i tworzenie jego historii

    history = model.fit(numpytraindata, numpytraintargets, validation_data=(numpyvaldata, numpyvaltargets), epochs=epochs, batch_size=1, callbacks=callbacks, verbose=1, shuffle=True)

#zapisanie modelu do pliku

model.save(f'{folderpath}{modelname}.h5')

#utworzenie grafów dotyczących statystyk oraz zapisanie ich wraz z modelem
#loss oznacza błąd średniokwadratowy

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.savefig(f'{folderpath}Loss per epoch.png')
plt.show()

accuracy_per_epoch = model.history.history['accuracy']
plt.plot(range(len(accuracy_per_epoch)),accuracy_per_epoch)
plt.savefig(f'{folderpath}Accuracy per epoch.png')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model Accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model Loss.png')
plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model mean absolute error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model mean absolute error.png')
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model root mean squared error')
plt.ylabel('Root mean squared error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model root mean squared error.png')
plt.show()

#test utworzonego modelu z wykorzystaniem bibliotek

test_loss, test_acc, test_mean_absolute_error, test_root_mean_square_error = model.evaluate(numpytestdata, numpytesttargets, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
print("Test MSE: {:.2f}".format(test_loss))
print("Test MAE: {:.2f}".format(test_mean_absolute_error))
print("Test RMSE: {:.2f}".format(test_root_mean_square_error))

