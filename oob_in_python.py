class Car:
    def __init__ (self, color, mileage):
        self.color = color
        self.mileage = mileage

    def __str__(self):
        return f"The {self.color} car has {self.mileage} miles."




def main():
    first_car = Car('blue', 20000)
    second_car = Car('red', 30000)

    print(first_car)
    print(second_car)



if __name__ == "__main__":
    main()