import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import xlsxwriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def draw_confusion_matrix(predicted_likes_groups, test_likes_groups, groups):
    cm = confusion_matrix(test_likes_groups, predicted_likes_groups)
    df_cm = pd.DataFrame(cm, groups, groups)
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.show()


def get_texts_and_likes_from_json(json_data):
    texts = []
    likes = []
    for item in json_data['items']:
        likes.append(item['likes']['count'])
        texts.append(item['text'])

    return texts, likes


def get_test_data_from_file(filename) -> (str, str):
    """
    Used for demonstration on test json
    :return:
    """
    with open(filename, "r", encoding='utf-8') as json_file:
        json_load = json.load(json_file)
    return get_texts_and_likes_from_json(json_load)


def get_text():
    with open('data/text.txt', "r", encoding='utf-8') as file:
        return file.read()


def data_from_skillbox_csv():
    df = pd.read_csv('data/vk_skillbox.csv')

    return df['text'].apply(str).tolist(), df['likes'].tolist()


def tf_idf_matrix_to_exel(terms, tf_idf_matrix):
    workbook = xlsxwriter.Workbook('data/matrix.xlsx')
    worksheet = workbook.add_worksheet('sheet')
    row = 0
    col = 0
    for word in tqdm(terms, desc='Writing to exel'):
        worksheet.write(row, col, word)
        col += 1
    for post in tf_idf_matrix:
        row += 1
        col = 0
        for word in post:
            worksheet.write(row, col, word)
            col += 1
    workbook.close()
