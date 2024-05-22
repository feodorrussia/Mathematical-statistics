import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from searchers.MNK import MNK
from searchers.Rabas import Rob

data_dir = "rawData"
number_of_row = 6


def filtering_data():
    def save_data_table():
        td = pd.DataFrame(table_info_for_saving)
        latex_output = td.to_latex()
        if not os.path.exists('tables'):
            os.makedirs('tables')
        # Записываем результат в файл
        with open(f'tables/table_with_v{input_voltage}.tex', 'w') as f:
            f.write(latex_output)

    avg_output_voltage = {}
    output_voltage = {}
    avg_sides_voltage = {}
    avg_sides_voltage_raw = {}
    avg_output_voltage_raw = {}
    data = []
    filtered_data = []
    table_info_for_saving = {"device": [],
                             "sigma": [],
                             "lbound with rule": [],
                             "average": [],
                             "upbound with rule": []
                             }
    past_voltage = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".dat"):
            input_voltage = filename.split("V")[0]

            if past_voltage != input_voltage:
                save_data_table()
                table_info_for_saving = {"device": [],
                                         "sigma": [],
                                         "lbound with rule": [],
                                         "average": [],
                                         "upbound with rule": [],
                                         }

            filepath = os.path.join(data_dir, filename)
            data = pd.read_csv(filepath, sep='\s+', header=None)

            if input_voltage not in avg_output_voltage:
                avg_output_voltage[input_voltage] = []
                avg_sides_voltage[input_voltage] = [[], []]
                avg_output_voltage_raw[input_voltage] = []
                avg_sides_voltage_raw[input_voltage] = [[], []]
                output_voltage[input_voltage] = []

            match = re.search('_sp(\d*)', filepath)
            number = match.group(1)

            mean = data[number_of_row].mean()

            avg_sides_voltage_raw[input_voltage][0].append(np.min(data[number_of_row]))
            avg_sides_voltage_raw[input_voltage][1].append(np.max(data[number_of_row]))

            std_dev = data[number_of_row].std()

            lower_bound = max(mean - 3 * std_dev, min(data[number_of_row]))
            upper_bound = min(mean + 3 * std_dev, max(data[number_of_row]))

            filtered_data = data[(data[number_of_row] >= lower_bound) & (data[number_of_row] <= upper_bound)]
            avg_output = filtered_data[number_of_row].mean()

            avg_output_voltage[input_voltage].append(avg_output)
            avg_output_voltage_raw[input_voltage].append(mean)

            output_voltage[input_voltage].append(filtered_data[number_of_row])

            avg_sides_voltage[input_voltage][0].append(lower_bound)
            avg_sides_voltage[input_voltage][1].append(upper_bound)

            table_info_for_saving["device"].append(number)
            table_info_for_saving["average"].append(mean)
            table_info_for_saving["sigma"].append(std_dev)
            table_info_for_saving["lbound with rule"].append(lower_bound)
            table_info_for_saving["upbound with rule"].append(upper_bound)
            past_voltage = input_voltage

    return avg_output_voltage, avg_sides_voltage, \
        data[number_of_row], filtered_data[number_of_row], output_voltage, avg_sides_voltage_raw, \
        avg_output_voltage_raw


def generate_data_for_liniar_regression_with_sigma(avg_output_voltage, avg_sides_voltage):
    avg_sides_voltage_m = []

    for input_voltage, output_voltages in avg_output_voltage.items():
        avg_output_voltage[input_voltage] = np.mean(output_voltages)
        avg_sides_voltage_m.append(
            (input_voltage, np.mean(avg_sides_voltage[input_voltage][0]), np.mean(avg_sides_voltage[input_voltage][1])))

    df = pd.DataFrame(list(avg_output_voltage.items()), columns=['Входное напряжение', 'Среднее выходное напряжение'])
    df['Входное напряжение'] = pd.to_numeric(df['Входное напряжение'])
    df.sort_values(by='Входное напряжение', inplace=True)

    df_sides = pd.DataFrame(avg_sides_voltage_m, columns=['Входное напряжение', "upper_side", "low_side"])
    df_sides['Входное напряжение'] = pd.to_numeric(df_sides['Входное напряжение'])
    df_sides.sort_values(by='Входное напряжение', inplace=True)

    return df, df_sides


def create_regression_graph(df):
    plt.plot(df['Входное напряжение'], df['Среднее выходное напряжение'], marker='o', linestyle='None', markersize=4)
    plt.xlabel('Input voltage (V)')
    plt.ylabel('Average output voltage')
    plt.title('Dependence of output voltage on input voltage')
    plt.grid(True)
    plt.show()


def create_regression_graph_(df, df_s, is_raw):
    def save_data():
        np.savetxt(f'line_regression_with_sigmas_sides_{("cl", "raw")[is_raw]}.csv',
                   np.column_stack((df['Входное напряжение'],
                                    df['Среднее выходное напряжение'],
                                    df_s['upper_side'], df_s['low_side'])),
                   delimiter=',', header='X, Y, Y_up, Y_low', comments='')

    def save_data_coef():
        print(f'MNK: {data["beta_hat"][0]}x + {data["alpha_hat"][0]}')
        print(f'Rabas: {data_r["beta_hat"][0]}x + {data_r["alpha_hat"][0]}')
        # TODO: Добавить сохранения data[betta_hat], data["alpha_hat"],
        #  data_r["beta_hat"][0], data_r["alpha_hat"][0]
        #  в виде таблицы для копирования в латех(инфа в txt).
        pass

    def print_graph():
        plt.plot(df['Входное напряжение'], df['Среднее выходное напряжение'], marker='o', linestyle='None',
                 markersize=4)
        plt.plot(df_s['Входное напряжение'], df_s['upper_side'], marker='o', linestyle='None', markersize=2,
                 color='red')
        plt.plot(df_s['Входное напряжение'], df_s['low_side'], marker='o', linestyle='None', markersize=2, color='red')
        plt.plot(x, y, color='blue')
        plt.plot(x, y_r, color='orange')
        plt.xlabel('Input voltage (V)')
        plt.ylabel('Average output voltage')
        plt.title('Dependence of output voltage on input voltage')
        plt.grid(True)
        plt.show()

    x = list(df['Входное напряжение'])
    x.extend(df_s['Входное напряжение'])
    x.extend(df_s['Входное напряжение'])

    y = list(df['Среднее выходное напряжение'])
    y.extend(df_s['upper_side'])
    y.extend(df_s['low_side'])

    data = MNK.get_coef([np.array(x), np.array(y)])
    data_r = Rob.get_coef([np.array(x), np.array(y)])

    x = np.linspace(min(df['Входное напряжение']), max(df['Входное напряжение']))
    y = data["beta_hat"][0] * x + data["alpha_hat"][0]
    y_r = data_r["beta_hat"][0] * x + data_r["alpha_hat"][0]

    save_data()
    save_data_coef()
    print_graph()


def create_hist(data, raw_flag):
    def save_data():
        bin_centers = (bins[:-1] + bins[1:]) / 2
        np.savetxt(f'histogram_data_for_latex_{("clean", "raw")[raw_flag]}.csv', np.column_stack((bin_centers, n)),
                   delimiter=',', header='Bin_Centers, Frequency', comments='')

    normalized_data = (data - np.mean(data)) / np.std(data)

    n, bins, patches = plt.hist(normalized_data,
                                bins=30,
                                density=True)

    save_data()
    # plt.close()
    # print_normalized_hist()
    # print_hist()


avg_output_voltage_, avg_sides_voltage_, \
    data_raw, filtered_data_, output_v, avg_sides_voltage_r, avg_output_voltage_r = filtering_data()

df_, df_sides = generate_data_for_liniar_regression_with_sigma(
    avg_output_voltage_r, avg_sides_voltage_r)

create_regression_graph_(df_, df_sides, is_raw=True)

df_, df_sides = generate_data_for_liniar_regression_with_sigma(
    avg_output_voltage_, avg_sides_voltage_)

create_regression_graph_(df_, df_sides, is_raw=False)

create_hist(filtered_data_, False)
create_hist(data_raw, True)
