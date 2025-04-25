import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" or __name__ == "Generator":
    from Square import Square
else:
    from .Square import Square


class Generator:
    def __init__(self, count: int, start_point: list[float, float] = [0, 0], strength: float = 1, speed: float = 1, max_weight: float = 100, max_price: float = 100, min_amount: int = 20, items_per_square: int = 10, **kwargs):
        self.count = count
        self.start_x, self.start_y = start_point

        self.strength = strength
        self.speed = speed
        self.max_weight = max_weight
        self.max_price = max_price
        self.min_amount = min_amount

        # self.items = items_per_square
        needed_squares = int(np.ceil(count / items_per_square))
        self.squares = []

        cols = int(np.ceil(np.sqrt(needed_squares)))
        rows = int(np.ceil(needed_squares / cols))

        for y in range(rows):
            for x in range(cols):
                if len(self.squares) >= needed_squares:
                    break
                self.squares.append(Square(x, y, items_per_square))

        self.area_size = len(self.squares)
        # print(self.area_size)

        self.dots = []
        self.max_bag_weight = 0

    def distance(self, position1: np.ndarray, position2: np.ndarray) -> float:
        return np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)

    def generate(self, **kwargs) -> np.ndarray:
        info = kwargs.get("info", False)
        for item in range(self.count):
            squares = self.squares.copy()
            not_passed = True
            while not_passed:
                square = squares[np.random.randint(0, len(squares))]
                x = np.random.uniform(square.x_min, square.x_max)
                y = np.random.uniform(square.y_min, square.y_max)
                distance = self.distance((x, y), (self.start_x, self.start_y))
                distance = distance / self.speed
                # weight = np.random.uniform(0.001, self.max_weight)
                weight = np.random.randint(1, self.max_weight+1)
                # multiplayer = self.strength / (distance * weight)
                # price = np.random.uniform(0.001, self.max_price) * multiplayer
                multiplayer = self.strength / (distance * weight)
                price = np.random.randint(1, self.max_price+1)
                if multiplayer > 1:
                    price += price*multiplayer
                else:
                    price -= price*multiplayer
                price = round(price)
                if price > self.max_price:
                    price = self.max_price
                elif price < 0:
                    price = 0
                try:
                    square.add_item(x, y, weight, price)
                    not_passed = False
                except Exception:
                    squares.remove(square)
                    # print(e)

        for square in self.squares:
            if len(square) > 0:
                if isinstance(self.dots, list):
                    self.dots = square.items
                else:
                    self.dots = np.append(self.dots, square.items, axis=0)

        prices = self.dots[:, 2]
        random_elements = np.random.choice(prices, size=self.min_amount if self.min_amount < self.count else self.count, replace=False)

        argsort = prices.argsort()
        minimum_global = self.dots[argsort, 2][:self.min_amount].sum()
        minimum = random_elements.sum()
        maximum = prices.sum()

        self.max_bag_weight = np.random.uniform(max(minimum, minimum_global), maximum)
        self.max_bag_weight = int(np.ceil(self.max_bag_weight))

        if info:
            print(self.dots[:, 2:], self.max_bag_weight)
        return self.max_bag_weight, self.dots[:, 2:]

    def plot(self):
        plt.title("Price")
        plt.grid(True)
        plt.scatter(self.start_x, self.start_y, c="Red", s=100, label="Start point")
        plt.scatter(self.dots[:, 0], self.dots[:, 1], c=self.dots[:, 3], cmap="copper", label="Items")
        plt.colorbar()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # for i in range(100):
    gen = Generator(10, [0, 0], strength=1, speed=1, max_weight=10, max_price=10, items_per_square=10, min_amount=5)
    gen.generate(info=True)
    # gen.plot()
