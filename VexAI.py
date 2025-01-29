import random

class VexAI:
    def __init__(self, text):
        self.words = text.split()
        self.vocab = list(set(self.words))
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.word_embeddings = {word: [random.random() for _ in range(10)] for word in self.vocab}

    def next_word(self, word):
        for i in range(len(self.words) - 1):
            if self.words[i] == word:
                return self.words[i + 1]
        return self.words[0]

    def sentence(self, word, length=4):
        sentence = word
        for _ in range(length - 1):
            word = self.next_word(word)
            sentence += ' ' + word
        return sentence

    def get_last_word(self, sentence):
        return sentence.split()[-1]

    def softmax(self, x):
        exp_x = [pow(2.718, i - max(x)) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]

    def matrix_multiplication(self, A, B):
        result = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
        return result

    def multi_head_attention(self, query, key, value, heads=2):
        d_k = len(query[0]) // heads
        query_heads = [query[i::heads] for i in range(heads)]
        key_heads = [key[i::heads] for i in range(heads)]
        value_heads = [value[i::heads] for i in range(heads)]
        
        scores = [[sum(q * k for q, k in zip(qh, kh)) / (d_k ** 0.5) for kh in key_heads] for qh in query_heads]
        attn_weights = [self.softmax(score) for score in scores]
        output = [[sum(a * v for a, v in zip(attn, vh)) for vh in value_heads] for attn in attn_weights]
        return [sum(o) for o in zip(*output)]

    def process_input(self, user_input):
        last_word = self.get_last_word(user_input)
        return self.sentence(last_word)
    
    def distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def distance_3d(x1, y1, z1, x2, y2, z2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    
    def distance_1d(x1, x2):
        return abs(x1 - x2)
    
    def midpoint(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def midpoint_3d(x1, y1, z1, x2, y2, z2):
        return ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
    
    def midpoint_1d(x1, x2):
        return (x1 + x2) / 2
    
    def slope(x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)
    
    def slope_3d(x1, y1, z1, x2, y2, z2):
        slopes = []
        if x2 != x1:
           slopes.append((x2 - x1) / (x2 - x1))
        else:
           slopes.append(None) 
        if y2 != y1:
           slopes.append((y2 - y1) / (y2 - y1))
        else:
          slopes.append(None)
        if z2 != z1:
           slopes.append((z2 - z1) / (z2 - z1))
        else:
           slopes.append(None)
    
        return slopes

    def slope_1d(x1, x2):
        return x2 - x1 
    
    def pathfind(x1, y1, x2, y2):
        path = []
        if x1 < x2:
            for i in range(x1, x2 + 1):
                path.append((i, y1))
        else:
            for i in range(x1, x2 - 1, -1):
                path.append((i, y1))
        if y1 < y2:
            for i in range(y1, y2 + 1):
                path.append((x2, i))
        else:
            for i in range(y1, y2 - 1, -1):
                path.append((x2, i))
        return path
    
    def triangulate_location(x1, y1, x2, y2, x3, y3):
        return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
    
    def triangulate_location_3d(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3
    
    def SLAM(initial_x, initial_y, AmmountMovedForward, AmmountMovedRight, AmmountMovedLeft, AmmountMovedBackward):
       new_x = initial_x + AmmountMovedRight - AmmountMovedLeft
       new_y = initial_y + AmmountMovedForward - AmmountMovedBackward
       return new_x, new_y
     
    def SLAM_3d(initial_x, initial_y, initial_z, AmmountMovedForward, AmmountMovedRight, AmmountMovedLeft, AmmountMovedBackward):
         new_x = initial_x + AmmountMovedRight - AmmountMovedLeft
         new_y = initial_y + AmmountMovedForward - AmmountMovedBackward
         new_z = initial_z
         return new_x, new_y, new_z
    
    def Vision(coordinates):
     grid_size = 10
     map_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
     for coord in coordinates:
        x, y = coord
        if 0 <= x < grid_size and 0 <= y < grid_size:
            map_grid[x][y] = 1

     obstacles = []
     for i in range(grid_size):
        for j in range(grid_size):
            if map_grid[i][j] == 0:
                obstacles.append((i, j))

     return map_grid, obstacles