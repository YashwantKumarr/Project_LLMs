Here is a complete, professional README file tailored to the fully advanced version of your code. It highlights not just what the application does, but all the impressive Python and software engineering concepts it demonstrates.

---

# 🏨 Advanced Hotel Management System (CLI)

## 📌 Overview

This project is a lightweight, Command-Line Interface (CLI) based Hotel Management System written in Python. Beyond its functional capabilities, this project serves as a comprehensive showcase of **Advanced Object-Oriented Programming (OOP)**, **Design Patterns**, and **Defensive Programming** principles. It features automated file-based data persistence and strict user input validation.

---

## ✨ Features

* **Dynamic Room Allocation:** Add and manage different room tiers (Standard, Suite) with unique pricing and tax behaviors.
* **Automated Data Persistence:** System state automatically saves to a local JSON database after every structural change.
* **Smart Validation:** Bulletproof input handling that prevents blank entries, rejects numerical values in guest names, and ensures prices/durations are positive.
* **Automated Billing:** Automatically calculates expected bills upon check-in and final invoices upon check-out.

---

## 🚀 Advanced Python Concepts Demonstrated

This codebase was engineered to mimic enterprise-level architecture. Key concepts implemented include:

### 1. Object-Oriented Programming (OOP)

* **Abstraction:** Utilizes the `abc` module (`ABC`, `@abstractmethod`) to create a `BaseRoom` blueprint that dictates necessary methods for all child classes.
* **Inheritance:** `StandardRoom` and `SuiteRoom` inherit core attributes from `BaseRoom` while maintaining their own specialized identities.
* **Polymorphism:** The `calculate_bill()` method is implemented differently in child classes (e.g., Suites automatically append a flat luxury tax, while Standard rooms do not).
* **Encapsulation:** Protects sensitive data like room pricing using private variables (`__price`) accessed safely via `@property` getters and `@price.setter` setters.

### 2. Design Patterns

* **Factory Pattern:** Implements a `RoomFactory` class to handle the logical creation of different room objects. The system passes parameters to the factory, which determines whether to return a `StandardRoom` or `SuiteRoom` object.

### 3. Advanced Python Features

* **Custom Decorators:** Implements an `@auto_save` decorator that wraps controller methods, cleanly executing data persistence without cluttering the business logic.
* **Generators:** Uses the `yield` keyword in `_generate_available_rooms()` for memory-efficient iteration over the active system directory.
* **Magic / Dunder Methods:** Overrides the `__str__` method to create clean, human-readable terminal outputs when printing complex objects.

### 4. Defensive Programming

* **Custom Exception Hierarchy:** Uses a base `HotelSystemException` inherited by highly specific custom errors (`InvalidInputException`, `RoomConflictException`, `RoomNotFoundException`, `BookingException`).
* **Input Sanitization:** Uses `any(char.isdigit() for char in guest_name)` to strictly enforce alphabetical name entries and `.strip()`/`.capitalize()` to standardize string formats.

---

## 🛠️ Built With

* **Python 3.x** (Standard Library Only - No external dependencies required)
* `json` - For state serialization.
* `uuid` - For generating secure, unique booking IDs.
* `datetime` - For check-in timestamping.
* `abc` - For Abstract Base Classes.

---

## 💻 How to Run

1. Ensure you have **Python 3.x** installed on your machine.
2. Clone or download this repository.
3. Open your terminal or command prompt and navigate to the directory containing the script.
4. Execute the script:
```bash
python hotel_manager.py

```


5. Follow the interactive menu on the screen.
6. *Note: A database file named `hotel_data.json` will be generated automatically in the same folder during your first session.*