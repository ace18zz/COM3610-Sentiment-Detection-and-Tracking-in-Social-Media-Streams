import matplotlib.pyplot as plt
from wordcloud import WordCloud
result = []
with open('./filter_txt_train_data.txt', encoding='utf-8') as fr:
    for line in fr:
        line_split = line.strip().split('\t')
        result.append(line_split[0])
result = ' '.join(result)
wc = WordCloud(
        background_color='white',
        width=500,
        height=350,
        max_words=1000,
        max_font_size=50,
        min_font_size=10,
        mode='RGBA'
        )
wc.generate(result)
wc.to_file(r"wordcloud.png")
plt.figure("jay")
plt.imshow(wc)
plt.axis("off")
plt.show()