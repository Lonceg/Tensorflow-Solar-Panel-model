#importowanie wymaganych bibliotek

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import Sequence

#utworzenie generatora szeregów czasowych z przesuwem okna co 24 godziny

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

#załadowanie pliku źródłowego

df = pd.read_csv('Dane wejściowe.csv', index_col='PeriodStart', parse_dates=True)

#dodanie dwóch kolumn
#kolumna dnia roku
#kolumna godziny dnia

df['day_of_year'] = df.index.dayofyear
df['hour_of_day'] = df.index.hour

#zmiennie wykorzystywane do szybkiej kontroli nad parametrami
#zmienna a oznacza rozmiar zestawu uczącego
#zmienna b oznacza rozmiar zestawu walidującego
#100-a-b oznacza rozmiar zestawu testowego
#n_input jest zmienną wprowadzaną do generatora gdzie odpowiada za ustalenie ilości godzin wykorzystywanych do wykonania predykcji 24 godzin
#(generator jest skonstruowany tak że po wpisaniu wartości większej niż 24, kolejne godziny ponad 24 będą z dni wcześniejszych)
#n_input informuje również strukturę sieci na temat rozmiaru wektora w warstwie wejściowej
#n_features jest zmienną opisującą ilość kolumn danych wejściowych wprowadzanych do sieci
#LSTM_nodes jest zmienną którą można szybko kontrolować ilość węzłów w każdej warstwie sieci, dla dokładniejszej kontroli można podmienić tę zmienną konkretnymi liczbami w sieci
#epochs oznacza ilość epok przez które trwała będzie nauka
#modelname oznacza to w jaki sposób utworzona zostanie nazwa pliku
#folderpath oznacza gdzie zostanie zapisany model wewnątrz folderu środowiska wirtualnego

a = 60
b = 30
n_input = 24
n_features = df.shape[1] - 1
LSTM_nodes = 256
epochs = 25
comment = f'{a},{b},{100-a-b}BILSTM2layer'
modelname = f'{n_input}Hours{LSTM_nodes}LSTM{epochs}Epochs{comment}'
folderpath = f'Modele/{modelname}/'

#wczytanie modelu wedle ścieżki

model = tf.keras.models.load_model(f'{folderpath}{modelname}.h5')

#przeliczanie rozmiarów danych wejściowych na podstawie zmiennych a,b

train_size = int(len(df) * a / 100)
val_size = int(len(df) * b / 100)
test_size = len(df) - train_size - val_size

#podział danych na zestawy uczące, walidujące oraz testowe

train_data = df[:train_size]
val_data = df[train_size:train_size+val_size]
test_data = df[train_size+val_size:]

#skalowanie zestawów wedle najmniejszej i największej wartości do przedziału między 0, a 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
resultsscaler = MinMaxScaler()

results = df.values
results = results[:, df.shape[1] - 3]
results = results.reshape(-1, 1)
scaler.fit(df)
resultsscaler.fit(results)

scaled_train = scaler.transform(train_data)
scaled_train_data_results = resultsscaler.transform(train_data.values[:, df.shape[1] - 3].reshape(-1, 1))
scaled_test = scaler.transform(test_data)
scaled_test_data_results = resultsscaler.transform(test_data.values[:, df.shape[1] - 3].reshape(-1, 1))
scaled_val = scaler.transform(val_data)
scaled_val_data_results = resultsscaler.transform(val_data.values[:, df.shape[1] - 3].reshape(-1, 1))

#wprowadzanie zestawów danych do generatów szeregów czasowych

generator_train_data = TimeSeriesGeneratorEvery24h(np.delete(scaled_train, df.shape[1] - 3, 1), scaled_train_data_results, length=n_input, batch_size=1)
generator_test = TimeSeriesGeneratorEvery24h(np.delete(scaled_test, df.shape[1] - 3, 1), scaled_test_data_results, length=n_input, batch_size=1)

#zamiana formatu z wykorzystaniem biblioteki numpy, niektóre funkcje lepiej współpracują z tym formatem

numpytraindata, numpytraintargets = convert_generator_to_numpy(generator_train_data)
numpytestdata, numpytesttargets = convert_generator_to_numpy(generator_test)

#obliczenie statystyk na podstawie funkcji zawartych w bibliotece tensorflow

test_loss, test_acc, test_mean_absolute_error, test_root_mean_square_error = model.evaluate(numpytestdata, numpytesttargets, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

rescaledMSE = resultsscaler.inverse_transform(np.reshape(test_loss, (1, -1)))[0][0]
rescaledMAE = resultsscaler.inverse_transform(np.reshape(test_mean_absolute_error, (1, -1)))[0][0]
rescaledRMSE = resultsscaler.inverse_transform(np.reshape(test_root_mean_square_error, (1, -1)))[0][0]
print("Scaled MSE (batch-wise calculation of fit()): {:.2f}".format(test_loss))
print("Scaled MAE: {:.2f}".format(test_mean_absolute_error))
print("Scaled RMSE: {:.2f}".format(test_root_mean_square_error))
print("Rescaled MSE: (batch-wise calculation of fit()): {:.2f}".format(rescaledMSE))
print("Rescaled MAE: {:.2f}".format(rescaledMAE))
print("Rescaled RMSE: {:.2f}".format(rescaledRMSE))
print("Rescaled RMSE as a percentage of peak power: {:.2f}%".format((rescaledRMSE/12.4)*100))

#obliczenie statystyk poprzez utworzenie pętli i ewaluowanie każdej pojedyńczej sekwencji na podstawie modelu bez wykorzystywania funkcji

y_actual_train = numpytraintargets
y_actual_test = numpytesttargets

y_pred_train = model.predict(numpytraindata)
y_pred_test = model.predict(numpytestdata)

timesteps = y_pred_train.shape[1]
sequences = y_pred_train.shape[0]
y_pred_train_reshaped = y_pred_train.reshape(sequences * timesteps, -1)

y_pred_train_reshaped = resultsscaler.inverse_transform(y_pred_train_reshaped)

y_actual_train_reshaped = y_actual_train.reshape(sequences * timesteps, -1)

y_actual_train_reshaped = resultsscaler.inverse_transform(y_actual_train_reshaped)

timesteps = y_pred_test.shape[1]
sequences = y_pred_test.shape[0]

y_pred_test_reshaped = y_pred_test.reshape(sequences * timesteps, -1)

y_pred_test_reshaped = resultsscaler.inverse_transform(y_pred_test_reshaped)

y_actual_test_reshaped = y_actual_test.reshape(sequences * timesteps, -1)

y_actual_test_reshaped = resultsscaler.inverse_transform(y_actual_test_reshaped)

#obliczenie pierwiastka błędu średniokwadratowego, średniego błędu bezwzględnego oraz błędu średniokwadratowego

mse = np.mean((y_actual_train_reshaped - y_pred_train_reshaped)**2)
mae = np.mean(np.abs(y_actual_train_reshaped - y_pred_train_reshaped))
rmse = np.sqrt(np.mean((y_actual_train_reshaped - y_pred_train_reshaped)**2))
max_power = 12.4
mse_percent = (mse / max_power**2) * 100
mae_percent = (mae / max_power) * 100
rmse_percent = (rmse / max_power) * 100

errors = y_pred_train_reshaped - y_actual_train_reshaped
std_dev = np.std(errors)
y_pred_train_reshaped = y_pred_train_reshaped[:, 0]

#utworzenie grafów

print("Calculated train MSE: {:.2f}".format(mse))
print("Calculated train MAE: {:.2f}".format(mae))
print("Calculated train RMSE: {:.2f}".format(rmse))
print("Calculated train MSE as a percentage of peak power: {:.2f}%".format(mse_percent))
print("Calculated train MAE as a percentage of peak power: {:.2f}%".format(mae_percent))
print("Calculated train RMSE as a percentage of peak power: {:.2f}%".format(rmse_percent))

df = pd.DataFrame(y_pred_train_reshaped)
df.to_csv('Prediction_train.csv', index=False)

plt.plot(y_actual_train_reshaped, label='Actual')
plt.plot(y_pred_train_reshaped, label='Predicted')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values Train')
plt.savefig(f'{folderpath}Actual vs Predicted Values Train.png')
plt.show()

mse = np.mean((y_actual_test_reshaped - y_pred_test_reshaped)**2)
mae = np.mean(np.abs(y_actual_test_reshaped - y_pred_test_reshaped))
rmse = np.sqrt(np.mean((y_actual_test_reshaped - y_pred_test_reshaped)**2))
max_power = 12.4
mse_percent = (mse / max_power**2) * 100
mae_percent = (mae / max_power) * 100
rmse_percent = (rmse / max_power) * 100

errors = y_pred_test_reshaped - y_actual_test_reshaped
std_dev = np.std(errors)
y_pred_test_reshaped = y_pred_test_reshaped[:, 0]

print("Calculated test MSE: {:.2f}".format(mse))
print("Calculated test MAE: {:.2f}".format(mae))
print("Calculated test RMSE: {:.2f}".format(rmse))
print("Calculated test MSE as a percentage of peak power: {:.2f}%".format(mse_percent))
print("Calculated test MAE as a percentage of peak power: {:.2f}%".format(mae_percent))
print("Calculated test RMSE as a percentage of peak power: {:.2f}%".format(rmse_percent))

df = pd.DataFrame(y_pred_test_reshaped)
df.to_csv('Prediction_test.csv', index=False)

plt.plot(y_actual_test_reshaped, label='Actual')
plt.plot(y_pred_test_reshaped, label='Predicted')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values Test')
plt.savefig(f'{folderpath}Actual vs Predicted Values Test.png')
plt.show()