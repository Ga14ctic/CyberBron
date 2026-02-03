# CyberBron Frontend Setup Guide

This guide provides detailed instructions for setting up and running the CyberBron frontend application.

## Prerequisites

- Node.js 16 or higher
- npm or yarn package manager
- CyberBron backend running on http://localhost:8000

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The application will be available at http://localhost:3000

### 3. Build for Production

```bash
npm run build
```

The build output will be in the `dist/` directory.

## Project Structure

```
frontend/
├── public/                  # Static assets
├── src/
│   ├── components/         # React components
│   │   ├── Auth/          # Authentication (Login, Register)
│   │   ├── Layout/        # Layout components (Navbar, Sidebar)
│   │   ├── Dashboard/     # Dashboard view
│   │   ├── Chat/          # Chat interface
│   │   ├── Notes/         # Notes management
│   │   ├── Flashcards/    # Flashcard study
│   │   └── Quiz/          # Quiz interface
│   ├── context/           # React contexts
│   │   └── AuthContext.tsx
│   ├── services/          # API service layer
│   │   ├── api.ts         # Axios instance with interceptors
│   │   ├── authService.ts
│   │   ├── chatService.ts
│   │   ├── notesService.ts
│   │   ├── flashcardsService.ts
│   │   └── quizService.ts
│   ├── types/             # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx            # Main app component with routing
│   ├── main.tsx           # Application entry point
│   └── index.css          # Global styles
├── index.html             # HTML template
├── package.json           # Dependencies and scripts
├── vite.config.ts         # Vite configuration
├── tsconfig.json          # TypeScript configuration
├── tailwind.config.js     # Tailwind CSS configuration
└── postcss.config.js      # PostCSS configuration
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Features

### Authentication
- JWT-based authentication
- Token stored in localStorage
- Auto-redirect on token expiration
- Protected routes

### Chat Interface
- Multiple chat sessions
- Real-time messaging with AI
- Markdown rendering for responses
- Session management (create, delete)
- Message history

### Notes System
- Create and edit notes
- Markdown support with preview
- Tag-based organization
- Search functionality
- CRUD operations

### Flashcards
- Create custom flashcards
- Spaced repetition algorithm
- Study mode with difficulty ratings
- Progress tracking
- Due card notifications

### Quizzes
- AI-generated quizzes
- Multiple choice questions
- Instant feedback
- Score tracking
- Review mode with explanations

### Dashboard
- Statistics overview
- Quick actions
- Recent activity feed
- Study streak tracking

## Configuration

### API Proxy

The development server proxies API requests to the backend:

```typescript
// vite.config.ts
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# API URL (optional, defaults to /api)
VITE_API_URL=http://localhost:8000/api
```

### Theme Customization

Customize the cybersecurity theme in `tailwind.config.js`:

```javascript
colors: {
  cyber: {
    primary: '#00ff88',      // Green accent
    secondary: '#00d4ff',    // Cyan accent
    dark: '#0a0e1a',        // Background
    darker: '#050810',      // Darker background
    gray: '#1a1f2e',        // Card background
    lightgray: '#2d3548',   // Borders
  },
}
```

## API Integration

### Authentication Flow

1. User logs in via `/login` endpoint
2. JWT token received and stored in localStorage
3. Token automatically added to all API requests via interceptor
4. On 401 error, token is cleared and user redirected to login

### API Service Example

```typescript
// Example: Fetching notes
import { notesService } from './services/notesService';

const notes = await notesService.getNotes();
```

All API services are located in `src/services/` and use the configured Axios instance.

## TypeScript Types

TypeScript types for API responses are defined in `src/types/index.ts`:

```typescript
interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

interface Note {
  id: number;
  user_id: number;
  title: string;
  content: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}
// ... more types
```

## Styling

### Tailwind CSS

The project uses Tailwind CSS with a custom cybersecurity theme:

- Custom color palette (cyber-primary, cyber-secondary, etc.)
- Custom animations (glow, pulse-slow)
- Responsive utilities
- Custom components (btn-primary, input-field, card)

### Global Styles

Global styles are defined in `src/index.css`:

- Custom scrollbar styling
- Base component classes
- Markdown content styling
- Loading spinner animation

## Development Tips

### Hot Module Replacement

Vite provides instant hot module replacement (HMR) during development. Changes to components will reflect immediately without page reload.

### TypeScript Strict Mode

The project uses TypeScript strict mode for better type safety. All components and services are fully typed.

### ESLint

Code quality is enforced with ESLint. Run `npm run lint` before committing changes.

### Component Structure

Follow this pattern for new components:

```typescript
import { useState, useEffect } from 'react';
import { SomeIcon } from 'lucide-react';

interface Props {
  // Component props
}

export default function ComponentName({ prop }: Props) {
  // Component logic
  
  return (
    <div className="card">
      {/* Component JSX */}
    </div>
  );
}
```

## Troubleshooting

### Port Already in Use

If port 3000 is already in use, modify `vite.config.ts`:

```typescript
server: {
  port: 3001, // Change to different port
}
```

### API Connection Issues

1. Ensure backend is running on http://localhost:8000
2. Check proxy configuration in `vite.config.ts`
3. Verify CORS is enabled in backend

### Build Errors

1. Clear node_modules and reinstall:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. Check TypeScript errors:
   ```bash
   npm run build
   ```

### Missing Types

If TypeScript complains about missing types, install them:

```bash
npm install --save-dev @types/package-name
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

### Build Size

Production build is optimized with:
- Code splitting
- Tree shaking
- Minification
- Gzip compression

Expected bundle size: ~410KB JS + ~18KB CSS (gzipped: ~126KB + ~4KB)

### Optimization Tips

1. Use lazy loading for routes:
   ```typescript
   const Dashboard = lazy(() => import('./components/Dashboard/Dashboard'));
   ```

2. Memoize expensive components:
   ```typescript
   const MemoizedComponent = memo(MyComponent);
   ```

3. Use React.memo for pure components
4. Implement virtual scrolling for long lists

## Deployment

### Build and Deploy

```bash
# Build for production
npm run build

# The dist/ folder can be deployed to any static hosting:
# - Vercel
# - Netlify
# - AWS S3 + CloudFront
# - GitHub Pages
# - Docker (see root Dockerfile)
```

### Docker Deployment

The project includes a multi-stage Dockerfile for containerized deployment. See the root `Dockerfile` and `docker-compose.yml` for details.

### Environment-specific Builds

For different environments, create separate `.env` files:

- `.env.development` - Development settings
- `.env.production` - Production settings
- `.env.staging` - Staging settings

## Contributing

When contributing to the frontend:

1. Follow the existing code style
2. Add TypeScript types for all functions and components
3. Ensure components are responsive
4. Add ARIA labels for accessibility
5. Test on multiple browsers
6. Run linter before committing
7. Update documentation if needed

## Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Guide](https://vitejs.dev/guide/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [React Router](https://reactrouter.com/)
- [Axios Documentation](https://axios-http.com/docs/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the backend API documentation
3. Check component README files
4. Open an issue on GitHub
