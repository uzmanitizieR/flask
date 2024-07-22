from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

data = pd.DataFrame({
    'bitki_turu': ['Domates', 'Salatalık', 'Biber', 'Patates', 'Patlıcan', 'Havuç', 'Limon', 'Portakal', 'Karpuz'],
    'gun_isigi': [12, 10, 9, 11, 8, 7, 10, 12, 11],
    'hava_sicakligi': [30, 28, 31, 29, 32, 27, 33, 32, 35],
    'azot': [2, 3, 1, 2, 3, 2, 1, 2, 4],
    'potasyum': [1, 2, 1, 2, 3, 1, 3, 2, 4],
    'kalsiyum': [2, 3, 2, 2, 3, 2, 1, 2, 4],
    'magnezyum': [1, 2, 2, 2, 3, 1, 3, 2, 4],
    'molibden': [1, 2, 1, 2, 3, 1, 3, 2, 4],
    'mangan': [2, 1, 3, 2, 2, 1, 3, 2, 4],
    'kükürt': [1, 2, 3, 2, 2, 1, 3, 2, 4],
    'demir': [1, 2, 1, 2, 3, 1, 3, 2, 4],
    'çinko': [2, 3, 1, 2, 3, 2, 1, 2, 4],
    'bakır': [1, 2, 3, 2, 2, 1, 3, 2, 4],
})

# Özellikler ve etiketleri ayırma
X = data[['gun_isigi', 'hava_sicakligi', 'azot', 'potasyum', 'kalsiyum', 'magnezyum', 'molibden', 'mangan', 'kükürt', 'demir', 'çinko', 'bakır']]
y = data['bitki_turu']

# One-Hot Encoding uygulama
X_encoded = pd.get_dummies(X, drop_first=True)

# Veriyi eğitim ve test setlerine böleme
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Random Forest sınıflandırma modelini oluşturma ve eğitme
bitki_model = RandomForestClassifier()
bitki_model.fit(X_train, y_train)


@app.route('/bitki_analizi')
def ana_sayfa():
    return render_template('bitki_analizi.html')


@app.route('/bitki_analizi', methods=['POST'])
def bitki_analiz():
    sonuc = None

    if request.method == 'POST':
        gun_isigi = int(request.form['gun_isigi'])
        hava_sicakligi = int(request.form['hava_sicakligi'])
        azot = int(request.form['azot'])
        potasyum = int(request.form['potasyum'])
        kalsiyum = int(request.form['kalsiyum'])
        magnezyum = int(request.form['magnezyum'])
        molibden = int(request.form['molibden'])
        mangan = int(request.form['mangan'])
        kükürt = int(request.form['kükürt'])
        demir = int(request.form['demir'])
        çinko = int(request.form['çinko'])
        bakır = int(request.form['bakır'])

        girdiler = [[gun_isigi, hava_sicakligi, azot, potasyum, kalsiyum, magnezyum, molibden, mangan, kükürt, demir, çinko, bakır]]
        tahmin = bitki_model.predict(girdiler)[0]

        sonuc = f"Önerilen Bitki Türü: {tahmin}"

    return render_template('bitki_analizi.html', sonuc=sonuc)






@app.route('/goruntuanaliz')
def goruntu_analizi_sayfasi():
    return render_template('goruntuanaliz.html')


@app.route('/analiz', methods=['POST'])
def analiz():
    sonuc = None

    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # Resmi yükle
            image = image.read()
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Görüntüyü gri tonlamada çevir
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Görüntüyü eşikleme ile işle
            _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # Eşikleme sonuçlarına göre temiz veya kirli suyu ayır
            white_pixel_count = np.sum(thresholded_image == 255)
            black_pixel_count = np.sum(thresholded_image == 0)

            if white_pixel_count > black_pixel_count:
                sonuc = "Temiz Su"
            else:
                sonuc = "Kirli Su"

    return render_template('goruntuanaliz.html', sonuc=sonuc)

@app.route('/nehir_analizi')
def nehir_analizi_sayfasi():
    return render_template('nehir_analizi.html')

@app.route('/nehir_analizi', methods=['POST'])
def nehir_analizi_sonuc():
    sonuc = "Nehir suyu içilebilir."

    elements = {
        "aluminium": 2.8,
        "ammonia": 32.5,
        "arsenic": 0.01,
        "barium": 2,
        "cadmium": 0.005,
        "chloramine": 4,
        "chromium": 0.1,
        "copper": 1.3,
        "flouride": 1.5,
        "bacteria": 0,
        "viruses": 0,
        "lead": 0.015,
        "nitrates": 10,
        "nitrites": 1,
        "mercury": 0.002,
        "perchlorate": 56,
        "radium": 5,
        "selenium": 0.5,
        "silver": 0.1,
        "uranium": 0.3
    }

    for element, limit in elements.items():
        value = float(request.form[element])
        if value > limit:
            sonuc = f"Nehir suyu arıtılmalıdır. {element} değeri tehlikeli düzeyde yüksektir."
            break

    return render_template('nehir_analizi.html', sonuc=sonuc)


@app.route('/', methods=['GET', 'POST'])
def su_kalitesi_analiz():
    sonuc = None

    if request.method == 'POST':
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])

        sonuc = su_kalitesi_analizi_yap(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity)

    return render_template('su_kalitesi_analiz.html', sonuc=sonuc)

def su_kalitesi_analizi_yap(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    if (6.5 <= ph <= 8.5 and
        hardness <= 250 and
        solids <= 500 and
        chloramines <= 4 and
        sulfate >= 3 and sulfate <= 30 and
        conductivity <= 400 and
        organic_carbon <= 2 and
        trihalomethanes <= 80 and
        turbidity <= 5):
        return "Suyunuz içilebilir."
    else:
        return "Suyunuz içilebilir değil."




if __name__ == '__main__':
    app.run(debug=True)
