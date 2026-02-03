import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, CreditCard, ClipboardCheck, MessageSquare, TrendingUp } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalNotes: 0,
    totalFlashcards: 0,
    totalQuizzes: 0,
    studyStreak: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // TODO: Fetch actual stats from API
        setStats({
          totalNotes: 12,
          totalFlashcards: 45,
          totalQuizzes: 8,
          studyStreak: 7,
        });
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const quickActions = [
    {
      title: 'Start Chat',
      description: 'Ask questions and learn with AI',
      icon: MessageSquare,
      link: '/chat',
      color: 'text-cyber-primary',
    },
    {
      title: 'Create Note',
      description: 'Take notes from your learning',
      icon: BookOpen,
      link: '/notes',
      color: 'text-cyber-secondary',
    },
    {
      title: 'Study Flashcards',
      description: 'Review your flashcards',
      icon: CreditCard,
      link: '/flashcards',
      color: 'text-purple-400',
    },
    {
      title: 'Take Quiz',
      description: 'Test your knowledge',
      icon: ClipboardCheck,
      link: '/quiz',
      color: 'text-yellow-400',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-cyber-primary mb-2">
          Welcome back, {user?.username}!
        </h1>
        <p className="text-gray-400">Ready to continue your cybersecurity journey?</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Total Notes</p>
              <p className="text-3xl font-bold text-cyber-primary">{stats.totalNotes}</p>
            </div>
            <BookOpen className="w-12 h-12 text-cyber-primary opacity-50" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Flashcards</p>
              <p className="text-3xl font-bold text-cyber-secondary">{stats.totalFlashcards}</p>
            </div>
            <CreditCard className="w-12 h-12 text-cyber-secondary opacity-50" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Quizzes Taken</p>
              <p className="text-3xl font-bold text-purple-400">{stats.totalQuizzes}</p>
            </div>
            <ClipboardCheck className="w-12 h-12 text-purple-400 opacity-50" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Study Streak</p>
              <p className="text-3xl font-bold text-yellow-400">{stats.studyStreak} days</p>
            </div>
            <TrendingUp className="w-12 h-12 text-yellow-400 opacity-50" />
          </div>
        </div>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-bold text-cyber-primary mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {quickActions.map((action) => (
            <Link
              key={action.title}
              to={action.link}
              className="card hover:border-cyber-primary transition-all group"
            >
              <action.icon className={`w-10 h-10 ${action.color} mb-3`} />
              <h3 className="text-lg font-semibold text-gray-100 mb-2 group-hover:text-cyber-primary transition-colors">
                {action.title}
              </h3>
              <p className="text-gray-400 text-sm">{action.description}</p>
            </Link>
          ))}
        </div>
      </div>

      <div className="card">
        <h2 className="text-2xl font-bold text-cyber-primary mb-4">Recent Activity</h2>
        <div className="space-y-4">
          <div className="flex items-start space-x-3 pb-4 border-b border-cyber-lightgray">
            <MessageSquare className="w-5 h-5 text-cyber-primary flex-shrink-0 mt-1" />
            <div>
              <p className="text-gray-300">Started a new chat session</p>
              <p className="text-gray-500 text-sm">2 hours ago</p>
            </div>
          </div>
          <div className="flex items-start space-x-3 pb-4 border-b border-cyber-lightgray">
            <BookOpen className="w-5 h-5 text-cyber-secondary flex-shrink-0 mt-1" />
            <div>
              <p className="text-gray-300">Created note: "Web Application Security"</p>
              <p className="text-gray-500 text-sm">5 hours ago</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <ClipboardCheck className="w-5 h-5 text-purple-400 flex-shrink-0 mt-1" />
            <div>
              <p className="text-gray-300">Completed quiz: "Network Security Basics"</p>
              <p className="text-gray-500 text-sm">1 day ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
