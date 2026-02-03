import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';
import Dashboard from './components/Dashboard/Dashboard';
import ChatInterface from './components/Chat/ChatInterface';
import NotesList from './components/Notes/NotesList';
import NoteEditor from './components/Notes/NoteEditor';
import FlashcardStudy from './components/Flashcards/FlashcardStudy';
import QuizTake from './components/Quiz/QuizTake';
import Navbar from './components/Layout/Navbar';
import Sidebar from './components/Layout/Sidebar';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-cyber-darker">
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 ml-64">
          {children}
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <AppLayout>
              <Dashboard />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <AppLayout>
              <ChatInterface />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/notes"
        element={
          <ProtectedRoute>
            <AppLayout>
              <NotesList />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/notes/:id"
        element={
          <ProtectedRoute>
            <AppLayout>
              <NoteEditor />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/flashcards"
        element={
          <ProtectedRoute>
            <AppLayout>
              <FlashcardStudy />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/quiz"
        element={
          <ProtectedRoute>
            <AppLayout>
              <QuizTake />
            </AppLayout>
          </ProtectedRoute>
        }
      />
      
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
