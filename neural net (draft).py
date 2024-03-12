import numpy as np

def get_unique_word_lengths(passage):
    words = passage.split()
    unique_words = list(set(word.lower() for word in words))
    unique_word_lengths = [len(word) for word in unique_words]
    return unique_words, len(unique_words), unique_word_lengths

passage = "Beneath the twilight sky, a radiant sun dipped, casting hues of orange and pink. A gentle breeze whispered through rustling leaves as day embraced the serenity, promising a night filled with dreams under the celestial canopy, adorned with stars that sparkled in the tranquil cosmic expanse."
check_unique_words, check_unique_word_count, input = get_unique_word_lengths(passage)

print(check_unique_words)
print(check_unique_word_count)
print(input)

weights = [np.random.rand(40) for _ in range(8)]
biases = [0.7, -0.5, 0.2, 0.8, 0.5, 0.1, 0.9, 0.3]

output = np.dot(weights, input) + biases
print(output)




