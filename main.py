import numpy as np

def get_unique_word_lengths(passage):
    words = passage.split()
    unique_words = list(set(word.lower() for word in words))
    unique_word_lengths = [len(word) for word in unique_words]
    return unique_words, len(unique_words), unique_word_lengths

passage_list = []
processed_passages = []
syllable_count = [5,8,5,10,6,7,3,10,8,4,7]

passages = ["Beneath the twilight sky, a radiant sun dipped, casting hues of orange and pink. A gentle breeze whispered through rustling leaves as day embraced the serenity, promising a night filled with dreams under the celestial canopy, adorned with stars that sparkled in the tranquil cosmic expanse.",
            "Amidst emerald meadows, butterflies pirouette, and wildflowers sway in a gentle cold breeze. Sunlight weaves through the brown leaves, creating a beautiful mosaic on the earth. A babbling brook murmurs tales, harmonizing with the birds' incessant melodies. Nature's divine canvas paints serenity, inviting tranquility.",
            "Upon a mountaintop, where air embraces, a lone eagle soars, wings spread in freedom. Valleys below cradle rivers that meander like liquid silver. Pine trees stand sentinel, their whispers carried by the wind. Nature's grandeur, an ode to the sublime, unfolds majestically.",
            "In the dense forest, beneath the canopy, a solitary traveler wandered, guided solely by his intuition. He marveled at the intricate tapestry of nature, where each leaf danced in harmony with the wind's whispers. The serenade enveloped him, soothing his restless soul as he journeyed, embracing the unknown.",
            "Beneath the velvet sky, the city pulsated with life, its streets illuminated by the neon glow of signs. People rushed past, lost in the rhythm of urban existence. Amidst the chaos, a lone musician played, his melodies weaving through the cacophony, offering solace to weary souls.",
            "In the big meadow, lots of colorful flowers bloomed, inviting butterflies to dance. Timmy giggled as he chased them, his laughter echoing through the air. The sun warmed his face, and he felt happy in nature's embrace, surrounded by beauty and joy, his heart fluttering with excitement.",
            "Within the tranquil grove, a symphony of bird songs filled the air, harmonizing with the rustle of leaves in the gentle breeze. Sarah strolled steadily along the long winding path, as her senses awash with the sights and sounds of nature's grandeur, finding priceless peace amidst life's chaos.",
            "Beneath the vast open star-strewn sky, the verdant expanse stretched endlessly, a worthy testament to nature's unrivaled resilience. Adam carelessly gazed upon the sprawling landscape, contemplating the intricate web of life that thrived within its bounds, humbled by the magnitude of existence unfolding before him.",
            "In the meadow, the lark sang melodies, while gentle zephyrs danced among the flowers. Sunlight cascaded, painting hues of gold on the verdant canvas. Nature's symphony played, evoking tranquil bliss. Each moment whispered secrets of life's intricate tapestry, weaving memories that lingered in the heart's sanctuary.",
            "In the marketplace, vendors called out their wares, their voices intermingling with the shoppers chatter. Aromas of spices and freshly baked bread wafted through the air. Colors dazzled in vibrant displays, enticing passersby to explore. Amidst the hustle and bustle, the heartbeat of the busy city pulsed lively with energy.",
            "In the city square, laughter echoed as people bustled about, their footsteps creating a rhythmic beat. Street performers captivated audiences with their talents, while vendors tempted passersby with smells. Amidst the urban symphony, the pulse of life thrived, weaving stories of diverse cultures and experiences.",
            ]

for passage in passages:
    check_unique_words, check_unique_word_count, input = get_unique_word_lengths(passage)
    #print(check_unique_word_count)
    passage_list.append(passage)
    processed_passages.append(input)

X = list(zip(passage_list, map(tuple, processed_passages), syllable_count))
#print(set(X))

y = X[0]
z = y[1]
test = list(z)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(40, 10)
layer1.forward(test)
print(layer1.output)
layer2 = Layer_Dense(10, 6)
layer2.forward(layer1.output)
print(layer2.output)