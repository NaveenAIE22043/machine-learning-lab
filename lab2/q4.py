def one_hot_encoding(cateogries):
    unique_cateogries=sorted(set(cateogries))
    encoding=[]
    for cateogry in cateogries:
        one_hot_vector=[0]*len(unique_cateogries)
        index=unique_cateogries.index(cateogry)
        one_hot_vector[index]=1
        encoding.append(one_hot_vector)

    return encoding
categories = ['red', 'blue', 'green', 'red', 'green', 'blue']

one_hot_encoded = one_hot_encoding(categories)
print("One-Hot Encoded:")
for category, one_hot_vector in zip(categories, one_hot_encoded):
    print(category, "->", one_hot_vector)