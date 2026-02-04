import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, CreditCard, ClipboardCheck, MessageSquare, TrendingUp, Target, Award, Calendar, BarChart3, FileText } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalNotes: 0,
    totalFlashcards: 0,
    totalQuizzes: 0,
    studyStreak: 0,
  });
  const [flashcardStats, setFlashcardStats] = useState({
    total_flashcards: 0,
    due_today: 0,
    mastered: 0,
    reviewed_today: 0,
    average_ease_factor: 2.5,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Fetch flashcard stats
        const token = localStorage.getItem('token');
        const response = await fetch('/api/flashcards/stats', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        
        if (response.ok) {
          const data = await response.json();
          setFlashcardStats(data);
          
          // Update stats with actual flashcard count
          setStats({
            totalNotes: 12,  // TODO: Fetch from API
            totalFlashcards: data.total_flashcards,
            totalQuizzes: 8,  // TODO: Fetch from API
            studyStreak: 7,  // TODO: Fetch from API
          });
        } else {
          // Fallback if API fails
          setStats({
            totalNotes: 12,
            totalFlashcards: 0,
            totalQuizzes: 8,
            studyStreak: 7,
          });
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        // Set fallback stats on error
        setStats({
          totalNotes: 12,
          totalFlashcards: 0,
          totalQuizzes: 8,
          studyStreak: 7,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const quickActions = [
    {
      title: 'View Notes',
      description: 'Your knowledge hub',
      icon: BookOpen,
      link: '/',
      color: 'text-cyber-primary',
    },
    {
      title: 'Chat with AI',
      description: 'Ask questions and learn',
      icon: MessageSquare,
      link: '/chat',
      color: 'text-cyber-secondary',
    },
    {
      title: 'Study Flashcards',
      description: 'Review your cards',
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
        <h1 className="text-3xl font-bold text-cyber-primary mb-2 flex items-center gap-3">
          <BarChart3 className="w-8 h-8" />
          Analytics & Insights
        </h1>
        <p className="text-gray-400">Track your learning progress and note activities</p>
      </div>

      {/* Note-Centric Stats */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-cyber-primary mb-4">Notes Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card bg-gradient-to-br from-cyber-gray to-cyber-lightgray">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Total Notes</p>
                <p className="text-3xl font-bold text-cyber-primary">{stats.totalNotes}</p>
                <p className="text-xs text-gray-500 mt-1">Your knowledge base</p>
              </div>
              <FileText className="w-12 h-12 text-cyber-primary opacity-50" />
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Generated Content</p>
                <p className="text-3xl font-bold text-cyber-secondary">{stats.totalFlashcards + stats.totalQuizzes}</p>
                <p className="text-xs text-gray-500 mt-1">From your notes</p>
              </div>
              <TrendingUp className="w-12 h-12 text-cyber-secondary opacity-50" />
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Flashcards</p>
                <p className="text-3xl font-bold text-purple-400">{stats.totalFlashcards}</p>
                <p className="text-xs text-gray-500 mt-1">Study materials</p>
              </div>
              <CreditCard className="w-12 h-12 text-purple-400 opacity-50" />
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Study Streak</p>
                <p className="text-3xl font-bold text-yellow-400">{stats.studyStreak} days</p>
                <p className="text-xs text-gray-500 mt-1">Keep it up!</p>
              </div>
              <TrendingUp className="w-12 h-12 text-yellow-400 opacity-50" />
            </div>
          </div>
        </div>
      </div>

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

      <div className="card bg-gradient-to-r from-cyber-gray to-cyber-lightgray mb-8">
        <h2 className="text-2xl font-bold text-cyber-primary mb-4">
          💡 Tip: Start with Notes
        </h2>
        <p className="text-gray-300 mb-4">
          Everything in CyberBron starts with your notes. Create comprehensive notes on your topics, 
          then use our AI tools to generate flashcards, quizzes, and presentations directly from your notes.
        </p>
        <Link to="/" className="btn-primary inline-flex items-center space-x-2">
          <BookOpen className="w-5 h-5" />
          <span>Go to Notes</span>
        </Link>
      </div>

      {/* Spaced Repetition Dashboard */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-cyber-primary mb-4 flex items-center gap-2">
          <Target className="w-7 h-7" />
          Spaced Repetition Progress
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card bg-gradient-to-br from-cyber-gray to-cyber-lightgray">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm mb-1">Due Today</p>
                <p className="text-4xl font-bold text-cyber-primary">{flashcardStats.due_today}</p>
                <p className="text-xs text-gray-400 mt-1">cards to review</p>
              </div>
              <Calendar className="w-12 h-12 text-cyber-primary opacity-40" />
            </div>
            {flashcardStats.due_today > 0 && (
              <Link to="/flashcards" className="mt-4 btn-primary text-sm w-full text-center block">
                Start Review Session
              </Link>
            )}
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Mastered</p>
                <p className="text-3xl font-bold text-green-400">{flashcardStats.mastered}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {flashcardStats.total_flashcards > 0 
                    ? `${Math.round((flashcardStats.mastered / flashcardStats.total_flashcards) * 100)}%` 
                    : '0%'}
                </p>
              </div>
              <Award className="w-12 h-12 text-green-400 opacity-50" />
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Reviewed Today</p>
                <p className="text-3xl font-bold text-blue-400">{flashcardStats.reviewed_today}</p>
                <p className="text-xs text-gray-500 mt-1">cards completed</p>
              </div>
              <CreditCard className="w-12 h-12 text-blue-400 opacity-50" />
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Avg. Ease Factor</p>
                <p className="text-3xl font-bold text-purple-400">{flashcardStats.average_ease_factor.toFixed(2)}</p>
                <p className="text-xs text-gray-500 mt-1">difficulty rating</p>
              </div>
              <TrendingUp className="w-12 h-12 text-purple-400 opacity-50" />
            </div>
          </div>
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
