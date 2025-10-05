# Context System: Chunking and Scoring Explained

This document explains how our context system breaks down code files and finds relevant information. Updated to reflect the current implementation with robust file handling, enhanced AST analysis, and performance optimizations.

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

Our system uses **"Enhanced Smart Chunking"** with robust file handling and advanced code structure analysis:

### Strategy 1: AST-Based Symbol Boundaries (Preferred)

We use Python's AST (Abstract Syntax Tree) to identify natural code boundaries:

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

# Chunk 4: Async functions are also detected
async def fetch_user_data(user_id: int):
    # async logic here
    pass
```

**What's new:**
- **Robust file reading** with multiple encoding support (UTF-8, Latin-1, CP1252)
- **File size limits** to prevent memory issues with large files
- **Async function detection** - includes `async def` functions in chunks
- **Enhanced error handling** for malformed or unreadable files

### Strategy 2: Smart Fallback (When AST Fails)

If AST parsing fails (syntax errors, binary files), we fall back to:
- Split by paragraph breaks (blank lines)
- Keep chunks under ~1200 characters
- Avoid cutting code in half
- **Safe directory traversal** with symlink detection and cycle prevention

## 📊 What is "Scoring"?

**Scoring** is like giving each chunk a "relevance grade" to answer: *"How well does this piece of code match what I'm looking for?"*

Think of it like a search engine - when you search for "pizza recipes", Google shows the most relevant results first, not random pages that happen to contain the word "pizza."

## 🎯 How Our Scoring Algorithm Works

Our scoring system uses a **BM25-like algorithm** with three main factors:

### 1. Token Matching with BM25 Scoring (The Foundation)

**What we do:** We use BM25 (Best Matching 25), an advanced text retrieval algorithm:

```
Search: "user registration"
Chunk A: "def register_user(email, password)" → Contains both "user" and "register" → Good match!
Chunk B: "def format_date(timestamp)" → Contains neither → Poor match
```

**Advanced scoring:**
- **Term Frequency (TF)**: How often does each search term appear in this chunk?
- **Inverse Document Frequency (IDF)**: How rare is each term across all files? Rare terms get bonus points
- **Document Length Normalization**: Prevents long chunks from dominating just because they have more words
- **Why this helps**: Combines the best of TF-IDF with document length considerations

### 2. Symbol Name Boost (+50% Bonus)

If your search matches a function or class name exactly, we give it a huge boost.

```
Search: "user validation"
Function named "validate_user" → Gets 1.5x score multiplier → Moves to top of results
```

**Why this works:** Function and class names are incredibly meaningful in code - if you're searching for "user validation" and there's a function called `validate_user`, it's almost certainly what you want.

### 3. Smart Tokenization (Enhanced Matching)

We use intelligent tokenization that understands code syntax:

```
Search: "validate_user" → Tokens: ["validate", "user"]
Search: "user-validation" → Tokens: ["user", "validation"]
Search: "validateUser" → Tokens: ["validate", "user"]
```

This catches variations in naming conventions (snake_case, camelCase, kebab-case).

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
- Encoding issues and large files can break simple text search

### Our Enhanced Solution Benefits:
- **Structure-Aware**: Uses AST parsing to understand functions, classes, and async functions
- **Relationship-Smart**: Finds connected pieces of code through imports and usage analysis
- **Robust File Handling**: Handles multiple encodings and large files gracefully
- **Performance Optimized**: Caching and early termination prevent resource waste
- **Error Resilient**: Graceful fallbacks when files are malformed or unreadable
- **Context-Rich**: Provides enough surrounding information to understand how pieces fit together
- **Precision-Focused**: Returns highly relevant results instead of keyword soup

### New Features:
- **Enhanced Import Analysis**: Preserves full module paths and handles relative imports
- **Safe Directory Traversal**: Detects symlinks and prevents infinite loops
- **Memory Management**: File size limits and cache management prevent memory issues
- **Comprehensive Error Handling**: Structured error reporting with recoverable vs fatal distinctions

## 🚀 The Result

When our test refinement system asks: *"What context do I need to fix this failing authentication test?"*

Instead of getting random code snippets, you get:
1. The exact `authenticate` function being tested (Score: 1.8) ⭐
2. The `User` class it depends on (Score: 1.2) ⭐
3. Related validation functions including async ones (Score: 0.9) ⭐
4. Documentation about auth flows (Score: 0.6)

### What Makes This Special:

- **Handles Edge Cases**: Works with files in different encodings, large files, and malformed syntax
- **Performance Optimized**: Caching prevents redundant processing of the same files
- **Memory Safe**: File size limits and smart chunking prevent memory exhaustion
- **Error Resilient**: Graceful degradation when encountering problematic files
- **Structure Aware**: Understands async functions, class methods, and import relationships

This gives the AI system exactly the right context to understand the problem and generate an accurate fix, even with complex, real-world codebases!

## 📝 Real-World Examples: From Code to Results

Let's walk through a concrete example to see how the context system processes real code:

### Example 1: User Authentication Module

**Input Code** (`auth.py`):
```python
from typing import Optional
from datetime import datetime, timedelta
import hashlib
import secrets

from database import Database
from models.user import User
from utils.crypto import hash_password
from exceptions.auth_errors import AuthenticationError

class AuthenticationService:
    def __init__(self, db: Database):
        self.db = db
        self.session_timeout = 3600  # 1 hour

    def register_user(self, email: str, password: str, name: str) -> User:
        """Register a new user with email and password."""
        if self.db.get_user_by_email(email):
            raise AuthenticationError("User already exists")

        hashed_password = hash_password(password)
        user = User(
            email=email,
            name=name,
            password_hash=hashed_password,
            created_at=datetime.utcnow()
        )

        return self.db.create_user(user)

    async def authenticate_user(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        user = self.db.get_user_by_email(email)
        if not user:
            return None

        if not self._verify_password(password, user.password_hash):
            return None

        # Generate session token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)

        self.db.create_session(user.id, token, expires_at)
        return token

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            hashed.encode('utf-8'),
            100000
        ).hex() == hashed
```

**How Chunking Works**:

The system creates these logical chunks:

```
Chunk 1: "AuthenticationService" (Score: 1.8)
├── class AuthenticationService:
├──     def __init__(self, db: Database):
├──         self.db = db
├──         self.session_timeout = 3600
├──
├──     def register_user(self, email: str, password: str, name: str) -> User:
├──         """Register a new user with email and password."""

Chunk 2: "register_user" (Score: 1.6)
├──         if self.db.get_user_by_email(email):
├──             raise AuthenticationError("User already exists")
├──
├──         hashed_password = hash_password(password)
├──         user = User(
├──             email=email,
├──             name=name,
├──             password_hash=hashed_password,
├──             created_at=datetime.utcnow()
├──         )
├──
├──         return self.db.create_user(user)

Chunk 3: "authenticate_user" (Score: 1.4)
├──     async def authenticate_user(self, email: str, password: str) -> Optional[str]:
├──         """Authenticate user and return session token."""
├──         user = self.db.get_user_by_email(email)
├──         if not user:
├──             return None
├──
├──         if not self._verify_password(password, user.password_hash):
├──             return None

Chunk 4: "_verify_password" (Score: 1.2)
├──     def _verify_password(self, password: str, hashed: str) -> bool:
├──         """Verify a password against its hash."""
├──         return hashlib.pbkdf2_hmac(
├──             'sha256',
├──             password.encode('utf-8'),
├──             hashed.encode('utf-8'),
├──             100000
├──         ).hex() == hashed
```

**Search Results** for query *"password validation"*:

| Rank | Chunk | Score | Why This Match |
|------|-------|-------|----------------|
| **#1** | `register_user` function | **1.6** | Contains password hashing and user creation logic |
| **#2** | `_verify_password` method | **1.2** | Direct password verification with secure hashing |
| **#3** | `authenticate_user` function | **0.8** | Uses password verification for authentication |
| **#4** | `AuthenticationService` class | **0.4** | Contains password-related methods |

### Example 2: API Route Handler

**Input Code** (`routes.py`):
```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.post import Post, PostCreate, PostResponse
from services.auth_service import get_current_user

router = APIRouter()

@router.post("/posts", response_model=PostResponse)
async def create_post(
    post_data: PostCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> PostResponse:
    """Create a new blog post."""
    try:
        new_post = Post(
            title=post_data.title,
            content=post_data.content,
            author_id=current_user["id"],
            published_at=datetime.utcnow() if post_data.published else None
        )

        db.add(new_post)
        db.commit()
        db.refresh(new_post)

        return PostResponse.from_orm(new_post)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create post")

@router.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: int,
    db: Session = Depends(get_db)
) -> PostResponse:
    """Get a specific post by ID."""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return PostResponse.from_orm(post)
```

**Search Results** for query *"create post endpoint"*:

| Rank | Chunk | Score | Why This Match |
|------|-------|-------|----------------|
| **#1** | `create_post` function | **2.1** | Perfect symbol match + contains "create" and "post" |
| **#2** | `get_post` function | **0.7** | Contains "post" but not "create" |
| **#3** | `APIRouter` configuration | **0.3** | Contains route definitions but no specific match |

### Example 3: Directory Structure Processing

**Input Directory Structure**:
```
src/
├── auth/
│   ├── __init__.py
│   ├── models.py
│   └── service.py
├── blog/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   └── templates/
├── database/
│   ├── __init__.py
│   └── connection.py
└── utils/
    ├── __init__.py
    ├── crypto.py
    └── validators.py
```

**Directory Summary Output**:
```
src/
├── auth/
│   ├── models.py (3 classes, 12 functions)
│   ├── service.py (5 classes, 18 functions)
│   └── __init__.py (2 functions)
├── blog/
│   ├── models.py (2 classes, 8 functions)
│   ├── routes.py (4 functions)
│   └── templates/ (3 files)
├── database/
│   ├── connection.py (3 classes, 5 functions)
│   └── __init__.py (1 function)
└── utils/
    ├── crypto.py (6 functions)
    ├── validators.py (4 functions)
    └── __init__.py (1 function)
```

**Key Insights**:
- **Smart filtering**: Skips `__pycache__`, `.git`, and other non-source directories
- **Structure awareness**: Groups related files and shows function/class counts
- **Safe traversal**: Handles permission errors and symlinks gracefully
- **Performance optimized**: Uses caching to avoid re-processing large directory trees
