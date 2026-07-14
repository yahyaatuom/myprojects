# The library management system rn shouldn't have all the features but minimal ones such as book, user and library access.

class Book:
    def __init__(self,id, title, author):
        self.id = id
        self.title = title
        self.author = author
        self._is_issued = False

    def issue(self):
        if self._is_issued:
            raise Exception(f"Book '{self.title}' is already issued")
        self._is_issued = True

    def return_book(self):
        if not self._is_issued:
            raise Exception(f"Book '{self.title} was not issued.")
        self._is_issued = False

    def is_available(self):
        return not self._is_issued

    def __repr__(self):
        status = "Available" if self._is_available() else "Issued"   
        return f"[{self.id}] {self.title} by {self.author} ({status})"
    

class User:
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.name = name
        self._issued_books = []

    def issue_book(self, book: Book):
        self._issued_books.append(book)

    def return_book(self, book: Book):
        self._issued_books.remove(book)

    def list_books(self):
        return self._issued_books
    
    def __repr__(self):
        return f"User({self.user_id}, {self.name})"
    
class Library:
    def __init__(self):
        self._books = {}
        self._users = {}

    #---- Book Management ----

    def add_book(self, book: Book):
        self._books[book.id] = book

    def get_book(self, id):
        return self._books.get(id)
    
    def list_books(self):
        return list(self._books.values())
    
    # === User Management ===

    def add_user(self, user: User):
        self._users[user.user_id] = user

    def get_user(self, user_id):
        return self._users.get(user_id)
    
    # //// core operations

    def issue_book(self, id, user_id):
        book = self.get_book(id)
        user = self.get_user(user_id)

        if not user or not book:
            raise Exception("Invalid book or user ID")
        
        if not book.is_available():
            raise Exception("The book has already been issued")
        
        book.issue()
        user.issue_book(book)

    def return_book(self, id, user_id):
        book = self.get_book(id)
        user = self.get_user(user_id)

        if not book or not user:
            raise Exception("Invalid book or user id")
        
        book.return_book()
        user.return_book(book)



if __name__ == "__main__":
    lib = Library()


    b1 = Book(1, "1984", "Yahya Khan")
    b2 = Book(2, "Clean Code", "Jamal Jackson")

    lib.add_book(b1)
    lib.add_book(b2)

    u1 = User(101, "ayesha")
    lib.add_user(u1)

    lib.issue_book(1, 101)

    print(lib.list_books())
    print(u1.list_books())
