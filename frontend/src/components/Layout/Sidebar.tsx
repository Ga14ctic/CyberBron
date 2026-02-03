import { NavLink } from 'react-router-dom';
import { Home, MessageSquare, BookOpen, CreditCard, ClipboardCheck } from 'lucide-react';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/notes', icon: BookOpen, label: 'Notes' },
  { to: '/flashcards', icon: CreditCard, label: 'Flashcards' },
  { to: '/quiz', icon: ClipboardCheck, label: 'Quiz' },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-cyber-gray border-r border-cyber-lightgray min-h-screen fixed left-0 top-16">
      <nav className="p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-4 py-3 rounded transition-all ${
                    isActive
                      ? 'bg-cyber-primary text-cyber-dark font-semibold'
                      : 'text-gray-300 hover:bg-cyber-lightgray hover:text-cyber-primary'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
