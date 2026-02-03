# CyberBron Frontend - Quick Reference

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Common Tasks

### Adding a New Component

1. Create component file in appropriate directory:
   ```typescript
   // src/components/Feature/MyComponent.tsx
   export default function MyComponent() {
     return <div>My Component</div>;
   }
   ```

2. Import and use in App.tsx or parent component

### Adding a New Route

In `src/App.tsx`:

```typescript
<Route
  path="/my-route"
  element={
    <ProtectedRoute>
      <AppLayout>
        <MyComponent />
      </AppLayout>
    </ProtectedRoute>
  }
/>
```

### Adding a New API Service

1. Create service file in `src/services/`:

```typescript
// src/services/myService.ts
import api from './api';
import { MyType } from '../types';

export const myService = {
  async getItems(): Promise<MyType[]> {
    const response = await api.get<MyType[]>('/my-endpoint');
    return response.data;
  },
};
```

2. Add types in `src/types/index.ts`

### Adding Sidebar Navigation

In `src/components/Layout/Sidebar.tsx`:

```typescript
const navItems = [
  // ... existing items
  { to: '/my-route', icon: MyIcon, label: 'My Feature' },
];
```

## Styling Reference

### Custom Colors

```typescript
// Use in className
className="text-cyber-primary"
className="bg-cyber-gray"
className="border-cyber-lightgray"
```

### Common Component Classes

```typescript
// Button
className="btn-primary"        // Primary action button
className="btn-secondary"      // Secondary button

// Input
className="input-field"        // Text input/textarea

// Card
className="card"               // Content card

// Effects
className="cyber-glow"         // Glowing text effect
className="loading-spinner"    // Loading animation
```

### Responsive Design

```typescript
// Mobile first approach
className="flex flex-col md:flex-row"  // Stack on mobile, row on desktop
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
```

## Common Patterns

### Loading State

```typescript
const [loading, setLoading] = useState(false);

if (loading) {
  return (
    <div className="flex items-center justify-center h-96">
      <div className="loading-spinner"></div>
    </div>
  );
}
```

### Error Handling

```typescript
const [error, setError] = useState('');

{error && (
  <div className="bg-red-900/20 border border-red-500 text-red-400 px-4 py-3 rounded">
    {error}
  </div>
)}
```

### API Call with Loading and Error

```typescript
const fetchData = async () => {
  setLoading(true);
  setError('');
  try {
    const data = await myService.getItems();
    setItems(data);
  } catch (err: any) {
    setError(err.response?.data?.detail || 'Failed to load data');
  } finally {
    setLoading(false);
  }
};
```

### Protected Route Access

```typescript
// Already handled by App.tsx
// Just use the route normally, authentication is automatic
```

### Using Auth Context

```typescript
import { useAuth } from '../context/AuthContext';

function MyComponent() {
  const { user, logout } = useAuth();
  
  return <div>Welcome, {user?.username}</div>;
}
```

## Icons

Using Lucide React icons:

```typescript
import { 
  Home, 
  MessageSquare, 
  BookOpen, 
  CreditCard,
  // ... more icons
} from 'lucide-react';

<Home className="w-5 h-5" />
```

[Browse all icons](https://lucide.dev/)

## Markdown Rendering

```typescript
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

<ReactMarkdown 
  remarkPlugins={[remarkGfm]} 
  className="prose prose-invert max-w-none"
>
  {markdownContent}
</ReactMarkdown>
```

## Form Handling

```typescript
const [formData, setFormData] = useState({ field1: '', field2: '' });

const handleSubmit = async (e: FormEvent) => {
  e.preventDefault();
  // Handle form submission
};

<form onSubmit={handleSubmit}>
  <input
    value={formData.field1}
    onChange={(e) => setFormData({ ...formData, field1: e.target.value })}
    className="input-field"
  />
</form>
```

## Navigation

```typescript
import { useNavigate, Link } from 'react-router-dom';

// Programmatic navigation
const navigate = useNavigate();
navigate('/my-route');

// Link component
<Link to="/my-route" className="text-cyber-primary">
  Go to My Route
</Link>
```

## TypeScript Tips

### Component Props

```typescript
interface Props {
  title: string;
  count?: number;  // Optional
  onAction: () => void;
}

export default function MyComponent({ title, count = 0, onAction }: Props) {
  // ...
}
```

### State with Types

```typescript
interface Item {
  id: number;
  name: string;
}

const [items, setItems] = useState<Item[]>([]);
```

## Common Issues & Solutions

### "Cannot find module" Error
```bash
npm install
```

### TypeScript Errors
```bash
npm run build  # Check all errors
```

### Styling Not Applied
- Check Tailwind class names are correct
- Restart dev server if adding new Tailwind classes

### API Calls Failing
- Check backend is running on http://localhost:8000
- Check API endpoint paths
- Check authentication token

### Build Fails
```bash
rm -rf node_modules dist
npm install
npm run build
```

## File Naming Conventions

- Components: PascalCase.tsx (e.g., `ChatInterface.tsx`)
- Services: camelCase.ts (e.g., `chatService.ts`)
- Types: index.ts in types directory
- Utils: camelCase.ts

## Component Organization

```
components/
  Feature/
    MainComponent.tsx      // Main feature component
    Subcomponent.tsx      // Supporting components
    index.ts              // Optional barrel export
```

## State Management

- **Local State**: useState for component-specific state
- **Context**: Auth context for global auth state
- **Props**: Pass data down component tree
- **API State**: Fetch and store in component state

## Best Practices

1. **Always use TypeScript types**
2. **Handle loading and error states**
3. **Add ARIA labels for accessibility**
4. **Use semantic HTML**
5. **Keep components small and focused**
6. **Extract repeated logic into custom hooks**
7. **Use memo for expensive renders**
8. **Clean up effects with return function**

## Useful Commands

```bash
# Find component usage
grep -r "ComponentName" src/

# Check bundle size
npm run build
ls -lh dist/assets/

# Update dependencies
npm update

# Check outdated packages
npm outdated
```

## Resources Quick Links

- [React Docs](https://react.dev/)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind](https://tailwindcss.com/)
- [Lucide Icons](https://lucide.dev/)
- [Vite](https://vitejs.dev/)
