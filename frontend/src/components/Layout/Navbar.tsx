import { useAuth } from '../../context/AuthContext';
import { Shield, LogOut, User, Plus, BookOpen } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const quickCreateNote = () => {
    navigate('/notes/new');
  };

  return (
    <nav className="bg-cyber-gray border-b border-cyber-lightgray sticky top-0 z-50">
      <div className="max-w-full mx-auto px-6">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <Shield className="w-8 h-8 text-cyber-primary" />
            <h1 className="text-2xl font-bold text-cyber-primary cyber-glow">
              CyberBron
            </h1>
            <span className="text-sm text-gray-400 hidden md:block ml-3">Notes-First Study Platform</span>
          </div>

          <div className="flex items-center space-x-4">
            {/* Quick Create Note Button */}
            <button
              onClick={quickCreateNote}
              className="btn-primary flex items-center gap-2 px-4 py-2"
              title="Quick Create Note"
            >
              <Plus className="w-4 h-4" />
              <BookOpen className="w-4 h-4" />
              <span className="hidden md:inline">New Note</span>
            </button>

            <div className="flex items-center space-x-2 text-gray-300">
              <User className="w-5 h-5" />
              <span>{user?.username}</span>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 text-gray-300 hover:text-cyber-primary transition-colors"
              aria-label="Logout"
            >
              <LogOut className="w-5 h-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
