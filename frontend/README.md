# CyberBron Frontend

Modern React + TypeScript frontend for the CyberBron AI-powered cybersecurity learning platform.

## Features

- 🎨 **Modern UI** - Built with React 18 and TypeScript
- 🎭 **Dark Theme** - Cybersecurity-inspired design with custom colors
- 🎨 **Tailwind CSS** - Utility-first CSS framework
- ⚡ **Vite** - Lightning-fast build tool and dev server
- 🔐 **Authentication** - JWT-based auth with protected routes
- 💬 **AI Chat** - Real-time chat interface with markdown support
- 📝 **Notes** - Create, edit, and organize notes with tags
- 🎴 **Flashcards** - Spaced repetition study system
- 📊 **Quizzes** - Take and generate AI-powered quizzes
- 📱 **Responsive** - Mobile-friendly design
- ♿ **Accessible** - ARIA labels and keyboard navigation

## Tech Stack

- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Markdown**: React Markdown with remark-gfm

## Getting Started

### Prerequisites

- Node.js 16+ and npm/yarn
- Backend API running on http://localhost:8000

### Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:3000

### Build for Production

```bash
npm run build
```

The optimized production build will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # React components
│   │   ├── Auth/       # Login and Registration
│   │   ├── Layout/     # Navbar and Sidebar
│   │   ├── Dashboard/  # Dashboard component
│   │   ├── Chat/       # Chat interface
│   │   ├── Notes/      # Notes management
│   │   ├── Flashcards/ # Flashcard study
│   │   └── Quiz/       # Quiz interface
│   ├── context/        # React contexts
│   │   └── AuthContext.tsx
│   ├── services/       # API service layer
│   │   ├── api.ts
│   │   ├── authService.ts
│   │   ├── chatService.ts
│   │   ├── notesService.ts
│   │   ├── flashcardsService.ts
│   │   └── quizService.ts
│   ├── types/          # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx         # Main app component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## Features Overview

### Authentication
- Login and registration forms
- JWT token management
- Protected routes
- Auto-redirect on token expiration

### Chat Interface
- Multiple chat sessions
- Real-time messaging
- Markdown rendering for AI responses
- Session management (create, delete)
- Auto-scroll to latest message

### Notes
- Create and edit notes with markdown support
- Tag system for organization
- Search functionality
- Preview mode
- Responsive card layout

### Flashcards
- Create custom flashcards
- Spaced repetition algorithm
- Study mode with difficulty ratings
- Progress tracking
- Due card notifications

### Quizzes
- AI-generated quizzes on any topic
- Multiple choice questions
- Instant feedback and explanations
- Score tracking
- Review mode

### Dashboard
- Statistics overview
- Quick actions
- Recent activity
- Study streak tracking

## Configuration

### API Proxy

The Vite dev server is configured to proxy API requests to the backend:

```typescript
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}
```

### Theme Customization

Customize colors in `tailwind.config.js`:

```javascript
colors: {
  cyber: {
    primary: '#00ff88',    // Green
    secondary: '#00d4ff',  // Cyan
    dark: '#0a0e1a',
    darker: '#050810',
    gray: '#1a1f2e',
    lightgray: '#2d3548',
  },
}
```

## Development

### Code Style

The project uses ESLint for code quality. Run linting:

```bash
npm run lint
```

### TypeScript

All components are written in TypeScript for type safety. Check types:

```bash
npm run build
```

## Environment Variables

Create a `.env` file if you need to customize the API URL:

```
VITE_API_URL=http://localhost:8000
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Contributing

1. Follow the existing code style
2. Add TypeScript types for new features
3. Ensure responsive design
4. Add ARIA labels for accessibility
5. Test on multiple browsers

## License

Part of the CyberBron project.
