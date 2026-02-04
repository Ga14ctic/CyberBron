import { NavLink } from 'react-router-dom';
import { MessageSquare, BookOpen, CreditCard, ClipboardCheck, Shield, BarChart3 } from 'lucide-react';

const navItems = [
  { to: '/', icon: BookOpen, label: 'Notes', highlight: true },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/flashcards', icon: CreditCard, label: 'Flashcards' },
  { to: '/quiz', icon: ClipboardCheck, label: 'Quiz' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/admin', icon: Shield, label: 'Admin' },
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
                  `flex items-center space-x-3 px-4 py-3 rounded transition-all duration-300 ${
                    isActive
                      ? 'bg-cyber-primary text-cyber-dark font-semibold shadow-lg shadow-cyber-primary/30'
                      : item.highlight
                      ? 'text-cyber-primary hover:bg-cyber-lightgray hover:text-cyber-primary hover:translate-x-1 font-medium'
                      : 'text-gray-300 hover:bg-cyber-lightgray hover:text-cyber-primary hover:translate-x-1'
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
