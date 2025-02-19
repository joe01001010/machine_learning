class Animals:
    def __init__(self, species, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight
        self.species = species

    def __str__(self):
        return f"The {self.species}'s name is {self.name}, they are {self.age} years old and weigh {self.weight} pounds."
    
    def eat(self, food = None):
        return f"{self.name} is eating." if food != None else f"{self.name} is not eating."
    
    def sleep(self):
        return f"{self.name} is sleeping."
    
    def move(self):
        return f"{self.name} is moving."
    
class Dog(Animals):
    def __init__(self, species, name, age, weight, breed):
        super().__init__(species, name, age, weight)
        self.breed = breed

    def bark(self):
        return f"{self.name} is barking."


class Turle(Animals):
    def __init__(self, species, name, age, weight, shell_color):
        super().__init__(species, name, age, weight)
        self.shell_color = shell_color

    def action(self):
        return f"{self.name} is running from Mario."
    
class GoldenRetriever(Dog):
    def __init__(self, species, name, age, weight, breed, color):
        super().__init__(species, name, age, weight, breed)
        self.color = color

    def bark(self):
        return f"The {self.species} named {self.name} is barking."


def main():
    first_dog = Dog('Dog', 'Rex', 3, 50, 'German Shepherd')
    second_dog = Dog('Dog', 'Max', 2, 40, 'Golden Retriever')
    first_turle = Turle('Turtle', 'Shelly', 5, 10, 'Green')
    second_turle = Turle('Turtle', 'Speedy', 3, 15, 'Brown')
    golden_retriever = GoldenRetriever('Dog', 'Buddy', 4, 60, 'Golden Retriever', 'Golden')

    print(golden_retriever)
    print(golden_retriever.bark())

    print(first_dog)
    print(second_dog)
    print(first_dog.eat())
    print(second_dog.sleep())
    print(first_dog.move())
    print(second_dog.bark())
    print()
    print(first_turle)
    print(second_turle)
    print(first_turle.eat())
    print(second_turle.sleep())
    print(first_turle.move())
    print(second_turle.action())


if __name__ == "__main__":
    main()