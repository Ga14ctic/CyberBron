"""
Example usage of CyberBron API endpoints
Run the backend server first: python -m backend.main
"""

import requests
import json

BASE_URL = "http://localhost:8000/api"

def example_auth():
    """Example: Register and login"""
    print("=" * 50)
    print("Authentication Example")
    print("=" * 50)
    
    # Register
    print("\n1. Registering new user...")
    response = requests.post(f"{BASE_URL}/auth/register", json={
        "email": "demo@cyberbron.com",
        "username": "demo_user",
        "password": "Demo123!@#",
        "full_name": "Demo User"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        token = data["access_token"]
        print(f"Token: {token[:30]}...")
        return token
    elif response.status_code == 400:
        print("User already exists, logging in...")
        
        # Login
        response = requests.post(f"{BASE_URL}/auth/login", json={
            "username": "demo_user",
            "password": "Demo123!@#"
        })
        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            print(f"Token: {token[:30]}...")
            return token
    
    print(f"Error: {response.json()}")
    return None


def example_notes(token):
    """Example: Create and manage notes"""
    print("\n" + "=" * 50)
    print("Notes Example")
    print("=" * 50)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a note
    print("\n1. Creating a note...")
    response = requests.post(f"{BASE_URL}/notes", 
        headers=headers,
        json={
            "title": "Python Basics",
            "content": "Python is an interpreted, high-level programming language.",
            "folder": "Programming",
            "tags": ["python", "programming", "basics"]
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        note = response.json()
        print(f"Created note ID: {note['id']}")
        note_id = note['id']
        
        # Get the note
        print(f"\n2. Retrieving note {note_id}...")
        response = requests.get(f"{BASE_URL}/notes/{note_id}", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Note: {response.json()['title']}")
        
        # Search notes
        print("\n3. Searching for 'Python'...")
        response = requests.get(f"{BASE_URL}/notes/search?q=Python", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {results['total']} notes")
        
        # List all notes
        print("\n4. Listing all notes...")
        response = requests.get(f"{BASE_URL}/notes?page=1&page_size=10", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Total notes: {results['total']}")


def example_flashcards(token):
    """Example: Create and review flashcards"""
    print("\n" + "=" * 50)
    print("Flashcards Example")
    print("=" * 50)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create flashcards
    print("\n1. Creating flashcards...")
    cards = [
        {
            "deck": "Python Fundamentals",
            "question": "What is a list in Python?",
            "answer": "A list is an ordered, mutable collection of items."
        },
        {
            "deck": "Python Fundamentals",
            "question": "What is the difference between append() and extend()?",
            "answer": "append() adds a single item, extend() adds multiple items from an iterable."
        }
    ]
    
    card_ids = []
    for card in cards:
        response = requests.post(f"{BASE_URL}/flashcards", headers=headers, json=card)
        if response.status_code == 201:
            data = response.json()
            card_ids.append(data['id'])
            print(f"Created flashcard ID: {data['id']}")
    
    # Get due flashcards
    print("\n2. Getting due flashcards...")
    response = requests.get(f"{BASE_URL}/flashcards/due?limit=5", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        due_cards = response.json()
        print(f"Due cards: {len(due_cards)}")
        
        # Review a card
        if due_cards and card_ids:
            print(f"\n3. Reviewing flashcard...")
            response = requests.post(
                f"{BASE_URL}/flashcards/{card_ids[0]}/review",
                headers=headers,
                json={
                    "card_id": card_ids[0],
                    "difficulty": "easy"
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Next review in {data['interval_days']} days")


def example_quiz(token):
    """Example: Create and take a quiz"""
    print("\n" + "=" * 50)
    print("Quiz Example")
    print("=" * 50)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a quiz
    print("\n1. Creating a quiz...")
    response = requests.post(f"{BASE_URL}/quizzes",
        headers=headers,
        json={
            "title": "Python Basics Quiz",
            "difficulty": "easy",
            "questions": [
                {
                    "type": "multiple_choice",
                    "question": "What is Python?",
                    "options": ["A snake", "A programming language", "A library", "A framework"],
                    "correct_answer": "A programming language",
                    "explanation": "Python is a high-level programming language."
                },
                {
                    "type": "true_false",
                    "question": "Python is compiled language",
                    "options": ["True", "False"],
                    "correct_answer": "False",
                    "explanation": "Python is an interpreted language."
                }
            ]
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        quiz = response.json()
        quiz_id = quiz['id']
        print(f"Created quiz ID: {quiz_id}")
        
        # Submit quiz attempt
        print(f"\n2. Submitting quiz attempt...")
        response = requests.post(
            f"{BASE_URL}/quizzes/{quiz_id}/attempt",
            headers=headers,
            json={
                "quiz_id": quiz_id,
                "answers": {
                    "0": "A programming language",
                    "1": "False"
                }
            }
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Score: {result['score']}/{result['max_score']} ({result['percentage']:.1f}%)")


def example_chat(token):
    """Example: Chat with AI"""
    print("\n" + "=" * 50)
    print("Chat Example")
    print("=" * 50)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Send a message
    print("\n1. Sending chat message...")
    response = requests.post(f"{BASE_URL}/chat",
        headers=headers,
        json={
            "message": "Explain Python lists in simple terms",
            "use_web_search": False
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        conversation_id = data['conversation_id']
        print(f"Conversation ID: {conversation_id}")
        print(f"Response: {data['message'][:100]}...")
        
        # List conversations
        print("\n2. Listing conversations...")
        response = requests.get(f"{BASE_URL}/conversations", headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            conversations = response.json()
            print(f"Total conversations: {len(conversations)}")


def main():
    """Run all examples"""
    print("\n" + "=" * 50)
    print("CyberBron API Examples")
    print("=" * 50)
    print("\nMake sure the backend server is running:")
    print("python -m backend.main")
    print("\n" + "=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=2)
        print(f"\n✓ Server is running (Status: {response.status_code})")
    except requests.exceptions.RequestException:
        print("\n✗ Server is not running!")
        print("Start it with: python -m backend.main")
        return
    
    # Run examples
    token = example_auth()
    if token:
        example_notes(token)
        example_flashcards(token)
        example_quiz(token)
        example_chat(token)
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        print("=" * 50)
        print("\nVisit http://localhost:8000/api/docs for interactive API documentation")


if __name__ == "__main__":
    main()
