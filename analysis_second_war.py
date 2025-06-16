import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Baixar recursos do NLTK se necessário
nltk.download("punkt")
nltk.download("stopwords")

st.title("Análise de Sentimento e Palavras Frequentes")

default_text = """
Fases e acontecimentos da Segunda Guerra Mundial
Podemos dividir a Segunda Guerra Mundial em duas fases:

Primeira fase (1939-1942)
Nessa fase, as tropas do Eixo avançaram rapidamente pela Europa. Em 1940, as tropas nazistas já ocupavam grande parte da França. Hitler fez questão de que a rendição francesa fosse assinada no mesmo vagão de trem que, em 1918, os alemães renderam-se logo após a derrota na Primeira Guerra Mundial. A Inglaterra foi atacada por aviões alemães. Em 1940, Winston Churchill foi eleito primeiro-ministro e iniciou a reação inglesa contra o ataque inimigo.

Essa fase favorável ao Eixo encerrou-se em 1941, quando as tropas nazistas foram derrotadas na União Soviética, após invasão ordenada por Hitler. Em dezembro do mesmo ano, os Estados Unidos foram atacados por kamikazes japoneses em sua base aérea de Pearl Harbor, no Oceano Pacífico. Os norte-americanos, com esse ataque, entraram na guerra.

Segunda fase (1943-1945)
A segunda fase da guerra foi definitiva para o término do conflito. Com a entrada dos Estados Unidos e da União Soviética no confronto, ingleses e franceses contaram com ajudas importantes para responder aos ataques nazifascistas. As tropas aliadas iniciaram o contra-ataque e reverteram o avanço do Eixo obtido na primeira fase. Do lado oriental, as tropas soviéticas; do lado ocidental, as tropas americanas, inglesas e francesas.

Na Europa, o Eixo foi perdendo espaço e sendo encurralado pelos Aliados. Benito Mussolini foi o primeiro líder a ser derrotado. Um dos dias mais marcantes para os Aliados na Segunda Guerra Mundial foi o dia 6 de junho de 1944, que entrou para a história como o Dia D. Nessa ocasião, ocorreu o desembarque dos aliados na Normandia, norte da França, ato que foi decisivo para encaminhar o Eixo à derrota ao iniciar a libertação francesa do domínio nazista.

A Itália foi o primeiro país do Eixo a se render, em 1943. Dois anos depois, veio a derrota nazista. Percebendo que a vitória dos Aliados era uma realidade, o Führer suicidou-se. Logo em seguida, os alemães renderam-se aos aliados, em 8 de maio de 1945. Esse dia foi comemorado como o Dia da Vitória. A Segunda Guerra na Europa já tinha terminado, mas, no Pacífico, os japoneses não assinaram a rendição e continuaram o combate, principalmente contra as tropas norte-americanas.

Bombas atômicas
A recusa do Japão em render-se e a vingança ao ataque a Pearl Harbor fizeram com que os Estados Unidos lançassem duas bombas atômicas nas cidades japonesas de Hiroshima, em 6 de agosto de 1945, e Nagasaki, dois dias depois. A destruição foi enorme e o imperador Hirohito não teve alternativa senão a rendição.

Brasil na Segunda Guerra Mundial
No começo da Segunda Guerra Mundial, o Brasil optou pela neutralidade. Getúlio Vargas governava o país como ditador desde 1937, quando deu o golpe do Estado Novo. Apesar da simpatia que ele e integrantes do governo tinham pelo nazifascismo, no primeiro momento, a neutralidade prevaleceu. O Brasil tinha acordos econômicos com potências europeias, e qualquer posicionamento brasileiro poderia comprometê-los.

A situação mudou a partir de 1942. O presidente norte-americano Franklin Roosevelt visitou o Brasil e teve um encontro com Vargas em Natal (RN). A base aérea da capital potiguar era estratégica para os aviões aliados deslocarem-se pelo Atlântico e atacarem o Eixo no norte da África e, em seguida, no sul europeu. O Brasil cederia a base aérea de Natal e, em troca, os Estados Unidos concederiam empréstimos para Vargas continuar sua política de investimento na indústria de base. Assim, o Brasil rompeu relações diplomáticas com os alemães e declarou guerra ao Eixo.

Ao contrário do que ocorreu na Primeira Guerra Mundial, o Brasil enviou tropas para a guerra na Europa. Em 1944 foi criada a Força Expedicionária Brasileira (FEB), que foi lutar contra as tropas nazistas na Itália. Apesar da rendição italiana no ano anterior, o país ainda tinha muitas tropas alemãs por lá. A participação da FEB foi vitoriosa, pois derrotou várias tropas inimigas. A vitória mais conhecida foi a conquista de Monte Castelo.
"""

user_text = st.text_area("Texto para análise:", value=default_text, height=400)

if st.button("Analisar Sentimento e Palavras"):
    # Análise de sentimento com TextBlob
    blob = TextBlob(user_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.subheader("Resultado com TextBlob:")
    st.write(f"Polaridade: {polarity:.2f} (de -1 negativo a +1 positivo)")
    st.write(f"Subjetividade: {subjectivity:.2f} (de 0 objetivo a 1 subjetivo)")

    # Análise de sentimento com transformers
    st.subheader("Resultado com modelo pré-treinado (transformers):")
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(user_text[:512])[
        0
    ]  # Limite de tokens para modelos base
    st.write(f"Label: {result['label']}")
    st.write(f"Score: {result['score']:.2f}")

    # Gráfico de palavras mais frequentes
    st.subheader("Palavras mais frequentes no texto:")

    # Tokenização e remoção de stopwords
    tokens = nltk.word_tokenize(user_text.lower())
    stop_words = set(stopwords.words("portuguese"))
    words = [word for word in tokens if word.isalpha() and word not in stop_words]

    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    df_freq = pd.DataFrame(most_common, columns=["Palavra", "Frequência"])

    st.bar_chart(df_freq.set_index("Palavra"))

    # WordCloud
    st.subheader("Nuvem de Palavras")
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stop_words,
        colormap="viridis",
    ).generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Interpretação simples
    if polarity > 0:
        st.success("O texto tem um sentimento geral POSITIVO.")
    elif polarity < 0:
        st.error("O texto tem um sentimento geral NEGATIVO.")
    else:
        st.info("O texto tem um sentimento geral NEUTRO.")

st.markdown("---")
st.caption(
    "Exemplo de análise de sentimento e visualização de palavras frequentes usando TextBlob, Transformers, NLTK, Matplotlib e WordCloud."
)
