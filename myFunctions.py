import pandas as pd
import random
from  datetime import datetime


# функция удаляет столбцы, выполняется после предподготовки данных
def filter_data(df_full: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'utm_keyword',
        'device_model',
        'client_id',
        'visit_number',
        'visit_date',
        'visit_time',
        'utm_source',
        'utm_medium',
        'device_screen_resolution',
        'geo_city'
    ]
    return df_full.drop(columns_to_drop, errors='ignore', axis=1)


# функция для заполнения пропусков  df_sessions
def prepare_missing(df_sessions: pd.DataFrame) -> pd.DataFrame:
    df_sessions = df_sessions.copy()

    # обработка пропусков в столбце utm_source
    df_sessions.utm_source = df_sessions.utm_source.fillna('ZpYIoDJMcFzVoPFsHGJL')

    # обработка значений в столбце utm_medium
    df_sessions['utm_medium'] = df_sessions.apply(
        lambda x: df_sessions.utm_medium.mode()[0] if x['utm_medium'] == '(not set)'
        else x['utm_medium'], axis=1)
    df_sessions['utm_medium'] = df_sessions.apply(lambda x: 'organic' if x['utm_medium'] == '(none)'
    else x['utm_medium'], axis=1)

    # обработка пропусков в столбце device_os
    android_mobile_device = list(set(df_sessions.device_brand.values))
    android_mobile_device.remove("Apple")

    df_sessions['device_os'] = df_sessions.apply(
        lambda x: 'Android' if pd.isna(x['device_os']) and x['device_category'] == 'mobile'
                               and (x['device_brand'] in android_mobile_device) else x['device_os'], axis=1)
    df_sessions['device_os'] = df_sessions.apply(
        lambda x: 'IOS' if pd.isna(x['device_os']) and x['device_category'] == 'mobile'
                           and (x['device_brand'] == 'Apple') else x['device_os'], axis=1)
    df_sessions['device_os'] = df_sessions.apply(lambda x: 'IOS' if x['device_os'] == 'iOS' else x['device_os'], axis=1)
    df_sessions['device_os'] = df_sessions.apply(lambda x: 'Apple' if pd.isna(x['device_os']) and
                                                                      (x['device_browser'] == 'Safari' or x[
                                                                          'device_browser'] == 'Safari (in-app)')
    else x['device_os'], axis=1)
    df_sessions['device_os'] = df_sessions.apply(
        lambda x: 'Windows' if pd.isna(x['device_os']) and (x['device_category'] == 'desktop')
        else x['device_os'], axis=1)
    df_sessions['device_os'] = df_sessions.apply(
        lambda x: 'Android' if pd.isna(x['device_os']) and (x['device_category'] == 'tablet')
        else x['device_os'], axis=1)

    os = list(((df_sessions.device_os.value_counts(dropna=False) / df_sessions.shape[0]).index)[:6])
    df_sessions['device_os'] = df_sessions.apply(lambda x: 'other' if x['device_os'] == '(not set)' or
                                                                      x['device_os'] not in os else x['device_os'],
                                                 axis=1)

    # обработка значений в столбце utm_campaign
    campaign = ['LTuZkdKfxRGVceoWkVyg', 'LEoPHuyFvzoNfnzGgfcd', 'FTjNLDyTrXaWYgZymFkV', 'gecBYcKZCPMcVYdSSzKP']
    df_sessions['utm_campaign'] = df_sessions.apply(lambda x: (random.choice(campaign)) if pd.isna(x['utm_campaign'])
    else x['utm_campaign'], axis=1)
    df_sessions['utm_campaign'] = df_sessions.apply(lambda x: 'other_campaign' if x['utm_campaign'] not in
                                                                                  ['LTuZkdKfxRGVceoWkVyg',
                                                                                   'LEoPHuyFvzoNfnzGgfcd',
                                                                                   'FTjNLDyTrXaWYgZymFkV',
                                                                                   'gecBYcKZCPMcVYdSSzKP']
    else x['utm_campaign'], axis=1)

    # обработка значений в столбце utm_adcontent
    df_sessions['utm_adcontent'] = df_sessions['utm_adcontent'].ffill()
    content = list(((df_sessions.utm_adcontent.value_counts(dropna=False) / df_sessions.shape[0]).index)[:4])
    df_sessions['utm_adcontent'] = df_sessions.apply(lambda x: 'other_content' if x['utm_adcontent']
                                                                                  not in content else x[
        'utm_adcontent'], axis=1)

    # обработка пропусков device_brand
    df_sessions['device_brand'] = df_sessions['device_brand'].fillna('other')
    df_sessions['device_brand'] = df_sessions.apply(lambda x: 'other' if x['device_brand'] == '(not set)'
    else x['device_brand'], axis=1)
    device = list(((df_sessions.device_brand.value_counts(dropna=False) / df_sessions.shape[0]).index)[:5])
    df_sessions['device_brand'] = df_sessions.apply(lambda x: 'other' if x['device_brand']
                                                                         not in device else x['device_brand'], axis=1)

    # обработка значений (not set) device_browser
    brouser_name = ['Chrome', 'Safari', 'Safari (in-app)', 'Android Webview', 'Samsung Internet', 'Opera', 'Firefox',
                    'Edge']

    df_sessions['device_browser'] = df_sessions.apply(lambda x: 'other' if x['device_browser'] not in
                                                                           brouser_name else x['device_browser'],
                                                      axis=1)

    # обработка значений (not set) geo_country
    df_sessions['geo_country'] = df_sessions.apply(
        lambda x: df_sessions.geo_country.mode()[0] if x['geo_country'] == '(not set)'
        else x['geo_country'], axis=1)
    df_sessions['geo_country'] = df_sessions.apply(
        lambda x: x['geo_country'] if x['geo_country'] == 'Russia' else 'other', axis=1)

    # обработка значений (not set) geo_city
    df_sessions['geo_city'] = df_sessions.apply(lambda x: 'other' if x['geo_city'] == '(not set)' else x['geo_city'],
                                                axis=1)

    return df_sessions


# функция обрабатывает признаки и создает новые из существующих для df_sessions
def create_feature(df_sessions: pd.DataFrame) -> pd.DataFrame:
    df_sessions = df_sessions.copy()

    # признаки на основе visit_date
    df_sessions['visit_date'] = pd.to_datetime(df_sessions.visit_date)
    df_sessions['day_of_the_week'] = df_sessions.visit_date.dt.day_name()
    df_sessions['quarter_of_the_year'] = df_sessions.visit_date.dt.quarter

    # признаки на основе visit_time
    def viewing_time(data):
        time = pd.to_datetime(data).strftime('%H:%M:%S')
        time = int(time.split(':')[0])
        if time > 0 and time <= 5:
            return 'night'
        elif time > 5 and time <= 10:
            return 'morning'
        elif time > 10 and time <= 18:
            return 'day'
        return 'evening'

    df_sessions['viewing_time'] = df_sessions.apply(lambda x: viewing_time(x['visit_time']), axis=1)

    # признаки на основе utm_source
    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxlTijuriZxsqZqt',
                    'ISrKoXQCxqqYvAZlCvjs', 'IZEXUFLARCUMynmHNBGo',
                    'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_sessions['advert_social_media'] = df_sessions.utm_source.apply(lambda x: 1 if x in social_media else 0)

    # признаки на основе utm_medium
    df_sessions['organic_traffic'] = df_sessions.apply(
        lambda x: 1 if x['utm_medium'] == 'organic' or x['utm_medium'] == 'referral' else 0, axis=1)

    # признаки на основе geo_city
    df_sessions['visit_big_city'] = df_sessions.apply(
        lambda x: 1 if x['geo_city'] == 'Moscow' or x['geo_city'] == 'Saint Petersburg' else 0, axis=1)
    moskow_region = ['Balashikha', 'Bronnitsy', 'Chernogolovka', 'Dmitrov', 'Dolgorudny', 'Domodedovo', 'Dubna',
                     'Elektrogorsk',
                     'Elektrostal', 'Fryazino', 'Ivanteevka', 'Kashira', 'Klimovsk', 'Kolomna', 'Korolev', 'Kotelniki',
                     'Krasnogorsk',
                     'Lytkarino', 'Lobnya', 'Losino-Petrovsky', 'Lukhovitsy', 'Lyubertsy', 'Mozhaysk', 'Mytishchi',
                     'Naro-Fominsk',
                     'Noginsk', 'Odintsovo', 'Orekhovo-Zuyevo', 'Pavlovsky Posad', 'Podolsk', 'Pushkino', 'Ramenskoye',
                     'Reutov',
                     'Rogovo', 'Ruza', 'Sergiyev Posad', 'Serpukhov', 'Solnechnogorsk', 'Stupino', 'Taldom', 'Vidnoye',
                     'Volokolamsk',
                     'Voskresensk', 'Yakhroma', 'Yegoryevsk', 'Zaraysk', 'Zvenigorod']
    df_sessions['moskow_area'] = df_sessions.apply(lambda x: 1 if x['geo_city'] in moskow_region else 0, axis=1)

    return df_sessions


# окончательная сборка датасета, определение целевой переменной
def prepare_full(df_sessions: pd.DataFrame, df_hits: pd.DataFrame) -> pd.DataFrame:
    target = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
              'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
              'sub_submit_success', 'sub_car_request_submit_click']
    df_hits['target_action'] = df_hits.apply(lambda x: 1 if x['event_action'] in target else 0, axis=1)
    target_action = df_hits.groupby('session_id')['target_action'].any().astype(int)
    df_full = pd.merge(left=df_sessions, right=target_action, on='session_id', how='inner')

    df_full = df_full.set_index('session_id')
    return df_full


def separation(df_sessions: pd.DataFrame, df_hits: pd.DataFrame):
    df = create_feature(prepare_missing(df_sessions))
    df_full = filter_data(prepare_full(df, df_hits))
    X = df_full.drop('target_action', axis=1)
    y = df_full['target_action']
    return X, y