def label_encode_categorical(data):
   
    unique_labels = list(set(data))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    encoded_data = [label_mapping[label] for label in data]

    return encoded_data, label_mapping

categorical_data = ["red", "blue", "green", "red", "blue", "green", "yellow"]
encoded_data, label_mapping = label_encode_categorical(categorical_data)

print("Original data:", categorical_data)
print("Encoded data:", encoded_data)
print("Label mapping:", label_mapping)

