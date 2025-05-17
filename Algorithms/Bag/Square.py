import numpy as np


class Square:
    def __init__(self, x_min: int, y_min: int, max_items_count: int):
        self.x_min = x_min
        self.x_max = x_min + 1
        self.y_min = y_min
        self.y_max = y_min + 1

        self.max_items_count = max_items_count
        # x, y, weight, price
        # self.items = np.zeros((max_items_count, 4))
        self.items = []

    def add_item(self, x: float, y: float, weight: float, price: float):

        if len(self.items) >= self.max_items_count:
            raise Exception("Square is full")

        if isinstance(self.items, list):
            self.items = np.array([[x, y, weight, price]])
        else:
            self.items = np.append(self.items, np.array([[x, y, weight, price]]), axis=0)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def __repr__(self):
        return str(self.items)


if __name__ == "__main__":
    s = Square(0, 0, 3)
    s.add_item(0.1, 0.2, 1, 2)
    s.add_item(0.1, 0.2, 2, 3)
    s.add_item(0.1, 0.2, 3, 4)
    s.add_item(0.1, 0.2, 4, 5)
    s.add_item(0.1, 0.2, 5, 6)
    s.add_item(0.1, 0.2, 6, 7)
    print(s)
