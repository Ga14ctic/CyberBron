# 🛡️ CyberBron Features Guide

## Quick Feature Overview

### 💬 Chat Tab - Intelligent AI Assistant

**What it does:**
- Answers cybersecurity questions using your documents, AI knowledge, and web search
- Automatically searches the web for current events, CVEs, and recent threats
- Shows where information comes from (📚 docs, 🧠 AI, 🌐 web)

**How to use:**
1. Type your question in the chat input
2. Get comprehensive answers with sources
3. Use quick action buttons:
   - 📝 Save to Notes - Save the response
   - 🎴 Make Flashcards - Generate study cards
   - 🎯 Create Slides - Generate a presentation

**Auto web search triggers:**
- Questions about "latest", "recent", "current" events
- CVE identifiers (CVE-2024-1234)
- Questions about 2024, 2025, 2026

---

### 📝 Notes Tab - Study Notes Management

**What it does:**
- Create and organize study notes
- Search across all your notes
- Export notes to Markdown
- Generate flashcards from notes

**How to use:**
1. Click "➕ New Note"
2. Enter title, content, tags, and folder
3. Use search bar to find notes
4. Filter by folder or tag
5. Edit, export, or delete notes

**Organization:**
- Folders: Group notes by topic (Network Security, Cryptography, etc.)
- Tags: Multiple tags per note for flexible categorization
- Search: Find notes by title or content

---

### 🎴 Flashcards Tab - Spaced Repetition Study

**What it does:**
- Create flashcards manually or with AI
- Study with spaced repetition algorithm
- Track your progress and mastery

**How to use:**

**Study Mode:**
1. Select a deck or "All Cards"
2. Choose "All Cards" or "Due for Review"
3. Click "Flip Card" to see the answer
4. Rate your knowledge:
   - 😄 Easy (review in 7 days)
   - 🙂 Medium (review in 3 days)
   - 😓 Hard (review in 1 day)

**Create Mode:**
- Manual: Enter question and answer
- AI Generate: Paste text, AI creates flashcards

**Decks Tab:**
- View deck statistics
- See all cards in each deck
- Track mastery progress

---

### 📊 Quiz Tab - Test Your Knowledge

**What it does:**
- Generate quizzes from study materials
- Take quizzes with instant grading
- Track your scores and progress

**How to use:**

**Take Quiz:**
1. Select a quiz
2. Click "▶️ Start Quiz"
3. Answer all questions
4. Submit for grading
5. Review detailed explanations

**Generate Quiz:**
1. Enter quiz title
2. Paste study content
3. Set number of questions (5-20)
4. Choose difficulty (easy/medium/hard)
5. AI generates quiz with explanations

**Question Types:**
- Multiple Choice (4 options)
- True/False
- Short Answer (AI graded)

**Results Tab:**
- View quiz history
- See average scores
- Track improvement

---

### 🎯 Presentations Tab - PowerPoint Generator

**What it does:**
- Generate professional PowerPoint presentations
- Multiple themes and customization
- Optional web research
- Download .pptx files

**How to use:**
1. Enter presentation topic
2. Choose content source:
   - Generate from Topic (AI creates content)
   - Use Custom Content (paste your notes)
3. Configure:
   - Number of slides (3-20)
   - Visual theme (Professional/Modern/Minimal/Dark)
   - Detail level (Brief/Moderate/Detailed)
4. Optional: Enable web research
5. Generate and download

**Themes:**
- **Professional**: Classic corporate style (blue/white)
- **Modern**: Contemporary design (light/gray)
- **Minimal**: Simple and clean (black/white)
- **Dark**: Cybersecurity theme (green/dark)

**Generated presentations include:**
- Title slide with date
- Content slides with bullet points
- Thank you slide
- Professional formatting

---

## 🎯 Quick Tips

### Chat Tips
- Ask specific questions for better answers
- Mention "latest" or "recent" to trigger web search
- Use quick actions to save good responses
- Check source indicators to verify information

### Notes Tips
- Use descriptive titles
- Add multiple tags for better searchability
- Organize by T-Level units (Unit 1, Unit 2, etc.)
- Export notes before exams for review

### Flashcards Tips
- Keep questions clear and concise
- Review cards daily for best retention
- Focus on cards marked "Hard"
- Generate from your worst quiz topics

### Quiz Tips
- Take quizzes regularly to track progress
- Review explanations for incorrect answers
- Generate quizzes from new topics you're learning
- Aim for consistent improvement

### Presentations Tips
- Enable web research for current topics
- Use Dark theme for cybersecurity presentations
- Generate 7-10 slides for 15-minute presentations
- Review and edit the generated content

---

## 🔧 Advanced Features

### Web Search Integration
- Automatically triggers for current events
- Searches cybersecurity-specific sources
- Shows results with citations
- Can be disabled in config.yaml

### Memory System
- Remembers your frequently studied topics
- Tracks your learning progress
- Stores your preferences
- Works across sessions

### Hybrid Knowledge Mode
- Uses your documents first
- Falls back to AI knowledge
- Searches web when needed
- Clearly indicates sources

### Source Citations
- 📚 From your documents
- 🧠 From AI knowledge
- 🌐 From web search
- Multiple sources combined

---

## 🎓 Study Workflow Examples

### Preparing for an Exam
1. Add textbook chapters to `data/`
2. Run `python ingest.py`
3. Chat: Ask questions about each topic
4. Save important responses to Notes
5. Generate flashcards from notes
6. Review flashcards daily
7. Take quizzes to test knowledge
8. Create presentation for group study

### Learning a New Topic
1. Chat: "Explain SQL injection attacks"
2. Save response to Notes
3. Generate flashcards
4. Take a quiz on the topic
5. Create presentation to share

### CVE Research
1. Chat: "What is CVE-2024-1234?"
2. Auto web search finds latest info
3. Save findings to Notes
4. Create flashcards for key points
5. Generate presentation for team

---

## ⌨️ Keyboard Shortcuts

While not all are available, here are some tips:
- **Enter** in chat input sends message
- **Ctrl+C** stops the server
- Use browser's **Ctrl+F** to search on page

---

## 🎨 UI Theme

**Color Scheme:**
- Primary: Cyber Green (#00ff88)
- Accent: Cyber Cyan (#00d4ff)
- Background: Dark (#0d1117)
- Text: Light Gray (#c9d1d9)

**Visual Style:**
- Dark mode for reduced eye strain
- Cybersecurity aesthetic
- Professional appearance
- Clear visual hierarchy

---

## 💡 Best Practices

### For T-Level Students
1. **Organize notes by unit** - Use folders for each T-Level unit
2. **Tag by topic** - Use consistent tags (network, crypto, threats)
3. **Daily review** - Study flashcards every day
4. **Weekly quizzes** - Test yourself once a week
5. **Share presentations** - Collaborate with classmates

### For Exam Prep
1. Generate flashcards from all topics
2. Take quizzes on weak areas
3. Review notes regularly
4. Use web search for latest threats
5. Create study presentations

### For Project Work
1. Research topics via chat
2. Organize findings in notes
3. Generate presentations
4. Create quizzes for team review
5. Export notes for documentation

---

**Need Help?** Check the README.md or TRANSFORMATION_SUMMARY.md for more details!
