# Context System: Chunking and Scoring Explained

This document explains how our context system breaks down code files and finds relevant information, using simple terms that don't require algorithm knowledge.

## 🧩 What is "Chunking"?

**Chunking** is like breaking a long book into chapters - instead of searching through entire files, we split them into smaller, meaningful pieces that are easier to work with.

### Why Do We Need Chunking?

Imagine you're looking for information about "user registration" in a 2000-line file. Without chunking:
- ❌ You'd get the entire file as one massive result
- ❌ Hard to find the specific part you need
- ❌ Overwhelming amount of irrelevant information

With chunking:
- ✅ You get just the `register_user` function 
- ✅ Clear, focused information
- ✅ Easy to understand what's relevant

## 🔨 How Our Chunking Works

Our system uses **"Smart Chunking"** that understands code structure:

### Strategy 1: Function and Class Boundaries (Preferred)

```python
# Original file gets split like this:

# Chunk 1: The User class
class User:
    def __init__(self, email: str):
        self.email = email

# Chunk 2: The register_user function  
def register_user(email: str, password: str):
    # validation logic here
    return User(email)

# Chunk 3: The validate_email function
def validate_email(email: str) -> bool:
    # email validation logic
    return "@" in email
```

**Why this works:** Each function or class becomes its own searchable piece with a clear name and purpose.

### Strategy 2: Smart Fallback (When Strategy 1 Fails)

If we can't find clear function/class boundaries, we fall back to:
- Split by paragraph breaks (blank lines)
- Keep chunks under ~1200 characters
- Avoid cutting sentences in half

## 📊 What is "Scoring"?

**Scoring** is like giving each chunk a "relevance grade" to answer: *"How well does this piece of code match what I'm looking for?"*

Think of it like a search engine - when you search for "pizza recipes", Google shows the most relevant results first, not random pages that happen to contain the word "pizza."

## 🎯 How Our Scoring Algorithm Works

Our scoring system considers three main factors:

### 1. Word Matching (The Foundation)

**Simple version:** Count how many words from your search appear in each chunk.

```
Search: "user registration" 
Chunk A: "def register_user(email, password)" → Contains both "user" and "register" → Good match!
Chunk B: "def format_date(timestamp)" → Contains neither → Poor match
```

**Advanced version:** We use TF-IDF (Term Frequency-Inverse Document Frequency):
- **Term Frequency**: How often does "user" appear in this chunk?
- **Inverse Document Frequency**: How rare is the word "user" across all files? (Rare words get bonus points)
- **Why this helps**: Common words like "the" don't dominate; specific words like "authenticate" get more weight

### 2. Symbol Name Boost (+50% Bonus)

If your search matches a function or class name exactly, we give it a huge boost.

```
Search: "user validation"
Function named "validate_user" → Gets 1.5x score multiplier → Moves to top of results
```

**Why this works:** Function names are incredibly meaningful in code - if you're searching for "user validation" and there's a function called `validate_user`, it's almost certainly what you want.

### 3. Length Normalization (Fairness Factor)

We prevent long chunks from winning just because they have more words.

```
Short chunk: "def login(user)" → 3 words, 1 match → High density
Long chunk: "def process_data_with_many_operations..." → 100 words, 1 match → Lower density
```

The short, focused chunk gets a higher score even though the long chunk might have more total matches.

## 🏆 Real-World Example: Scoring in Action

**Search Query:** "password validation"

| Chunk | Content | Score | Why? |
|-------|---------|-------|------|
| **#1** | `def validate_password(password: str) -> bool:` | **1.6** | Perfect symbol match + content relevance |
| **#2** | `# Password requirements: 8+ chars, special symbols` | **0.8** | Good content match, no symbol |
| **#3** | `def register_user(email, password):` | **0.3** | Contains "password" but not main focus |
| **#4** | `def format_output(data):` | **0.0** | No relevant terms |

## 📈 Score Ranges Guide

| Score | Meaning | When You'll See This |
|-------|---------|---------------------|
| **1.5+** | Excellent match | Function name matches your search + relevant content |
| **0.8-1.5** | Good match | Several search terms found, contextually relevant |
| **0.3-0.8** | Fair match | Some search terms found, partially relevant |
| **0.1-0.3** | Weak match | Few matching words, likely not what you want |
| **0.0** | No match | No search terms found |

## 🔧 Why This Approach Works Well for Code

### Traditional Search Problems with Code:
- File names don't always match content (`utils.py` could contain anything)
- Important information scattered across multiple files
- Need to understand relationships between different pieces

### Our Solution Benefits:
- **Structure-Aware**: Understands functions, classes, and their purposes
- **Relationship-Smart**: Finds connected pieces of code through imports and usage
- **Context-Rich**: Provides enough surrounding information to understand how pieces fit together
- **Precision-Focused**: Returns highly relevant results instead of keyword soup

## 🚀 The Result

When our test refinement system asks: *"What context do I need to fix this failing authentication test?"*

Instead of getting random code snippets, you get:
1. The exact `authenticate` function being tested (Score: 1.8)
2. The `User` class it depends on (Score: 1.2) 
3. Related validation functions (Score: 0.9)
4. Documentation about auth flows (Score: 0.6)

This gives the AI system exactly the right context to understand the problem and generate an accurate fix!
