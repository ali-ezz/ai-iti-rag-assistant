# LAB-01-OOP

Source notebook: LAB-01-OOP.ipynb

# Day 1 Lab — Object-Oriented Programming
## Hands-On Practice Lab

This lab is **separate from the main project**. It contains a large set of progressively harder exercises to reinforce today's OOP concepts: classes, encapsulation, inheritance, polymorphism, dunder methods, and class/static methods.

Work through each exercise in order. Solution cells are provided below each task — try to solve it yourself first before checking.

### Structure
- Part A: Classes & Objects Basics (Exercises 1–4)
- Part B: Encapsulation (Exercises 5–7)
- Part C: Inheritance & Polymorphism (Exercises 8–11)
- Part D: Dunder Methods (Exercises 12–14)
- Part E: Class Methods & Static Methods (Exercises 15–16)
- Part F: Mini Challenges — Combine Everything (Exercises 17–20)

## Part A — Classes & Objects Basics

### Exercise 1: `Book` Class
Create a class `Book` with attributes `title`, `author`, and `price`. Add a method `discount(percent)` that reduces the price by the given percentage and prints the new price.

### Exercise 2: `Rectangle` Class
Create a `Rectangle` class with `width` and `height`. Add methods `area()`, `perimeter()`, and `is_square()` (returns `True` if width equals height).

### Exercise 3: `Playlist` Class With a Running Total
Create a `Playlist` class that stores songs (each with a `name` and `duration` in seconds). Add methods `add_song(name, duration)` and `total_duration()` that returns total time formatted as `MM:SS`.

### Exercise 4: Class Attribute — `id` Auto-Increment
Create a `Ticket` class where every new instance automatically gets a unique, incrementing `ticket_id`, starting from 1000. Use a class attribute to track the next available ID.

## Part B — Encapsulation

### Exercise 5: Protected `Password` Field
Create a `User` class with a public `username` and a protected `_password`. Add a method `check_password(attempt)` returning `True`/`False`.

### Exercise 6: Private Balance with Validation
Create a `Wallet` class with a private `__balance`. Add `deposit(amount)` and `withdraw(amount)` methods that reject negative amounts or overdrafts, printing an appropriate error message.

### Exercise 7: `@property` for a Validated Age
Create a `Person` class where `age` is stored privately, exposed through a `@property`, and the setter raises `ValueError` if age is negative or greater than 150.

## Part C — Inheritance & Polymorphism

### Exercise 8: `Shape` Hierarchy
Create a base class `Shape` with a method `area()` that raises `NotImplementedError`. Create `Circle` and `Rectangle` subclasses that implement `area()` properly.

### Exercise 9: Employee Hierarchy With `super()`
Create a base class `Employee` with `name` and `base_salary`. Create a `Manager` subclass that adds a `bonus` and overrides a `total_pay()` method (base does `total_pay()` returning `base_salary`; `Manager` adds the bonus on top using `super()`).

### Exercise 10: Notification System (Polymorphism)
Create a base class `Notification` with a `send(message)` method. Create `EmailNotification` and `SMSNotification` subclasses that override `send()` differently. Write a function `notify_all(notifications, message)` that sends the same message through a list of different notification types.

### Exercise 11: Abstract-Style Base with `ABC`
Use Python's `abc` module to make `Shape` a true abstract base class, so it cannot be instantiated directly and subclasses are forced to implement `area()`.

## Part D — Dunder Methods

### Exercise 12: `Money` Class With Arithmetic Operators
Create a `Money` class with an `amount` and `currency`. Implement `__add__`, `__sub__`, `__str__`, and `__eq__` so two `Money` objects (same currency) can be added, subtracted, compared, and printed nicely.

### Exercise 13: `Deck` Class With `__len__` and `__getitem__`
Create a `Deck` class holding a list of card strings. Implement `__len__` and `__getitem__` so you can call `len(deck)`, index into it (`deck[0]`), and loop over it directly with a `for` loop.

### Exercise 14: `__repr__` vs `__str__`
Create a `Point` class with `x` and `y`. Implement both `__str__` (a friendly display, e.g. `"(3, 4)"`) and `__repr__` (an unambiguous developer-facing form, e.g. `"Point(x=3, y=4)"`). Demonstrate the difference between `print(p)` and just evaluating `p` in a cell / calling `repr(p)`.

## Part E — Class Methods & Static Methods

### Exercise 15: `Date` Class With Alternative Constructors
Create a `Date` class with `day`, `month`, `year`. Add a class method `from_string(date_string)` that parses a `"DD-MM-YYYY"` string, and a static method `is_leap_year(year)`.

### Exercise 16: Singleton-Style Configuration Using Class Methods
Create a `Config` class that stores app-wide settings in a **class attribute** dictionary, with class methods `set(key, value)` and `get(key)` — so all parts of a program share the same configuration without creating instances.

## Part F — Mini Challenges (Combine Everything)

### Exercise 17: Library System
Build a small library system:
- `Book` class: `title`, `author`, `is_borrowed` (default `False`)
- `Library` class: holds a list of `Book` objects, with methods `add_book`, `borrow_book(title)`, `return_book(title)`, and `list_available_books()`

### Exercise 18: Employee Hierarchy With Polymorphic Payroll
Build `Employee` (base), `HourlyEmployee` (paid by `hours_worked * hourly_rate`), and `SalariedEmployee` (fixed `monthly_salary`). Each overrides a `calculate_pay()` method. Write a `run_payroll(employees)` function that prints total payroll using polymorphism.

### Exercise 19: Custom Exception + Encapsulated Inventory System
Create a custom exception `OutOfStockError`. Build an `InventoryItem` class with a private `__quantity`. Add `sell(amount)` that raises `OutOfStockError` if there isn't enough quantity, and `restock(amount)`.

### Exercise 20: Final Challenge — A Small Game Character System
Build a small combat system:
- Base class `Character` with `name`, `health`, `attack_power`. Methods: `attack(other)` (reduces `other.health`), `is_alive()`, `__str__`.
- Subclasses `Warrior` (extra `defense` that reduces incoming damage) and `Mage` (extra `mana`, and a `cast_spell(other)` method that does double damage but costs mana).
- Simulate a short battle between a `Warrior` and a `Mage` using a loop, printing each round's result until one side's health drops to 0 or below.

---
## Lab Wrap-Up

You've now practiced:
- Building classes from scratch with meaningful constructors
- Protecting internal state with encapsulation and `@property`
- Reusing and extending behavior with inheritance and `super()`
- Writing polymorphic functions that work across different object types
- Customizing how your objects behave with dunder methods
- Using class methods as alternative constructors and static methods as utilities
- Combining everything into small realistic systems (library, payroll, inventory, and a game)

Keep this lab as a reference — you'll build on these same patterns throughout the rest of the course.
