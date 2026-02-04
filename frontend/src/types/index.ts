export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Message {
  id: number;
  session_id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ChatSession {
  id: number;
  user_id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages?: Message[];
}

export interface Note {
  id: number;
  user_id: number;
  title: string;
  content: string;
  tags: string[];
  folder: string;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface Flashcard {
  id: number;
  user_id: number;
  front: string;
  back: string;
  difficulty: 'easy' | 'medium' | 'hard';
  last_reviewed?: string;
  next_review?: string;
  created_at: string;
}

export interface Quiz {
  id: number;
  user_id: number;
  title: string;
  description?: string;
  questions: QuizQuestion[];
  score?: number;
  created_at: string;
  completed_at?: string;
}

export interface QuizQuestion {
  id: number;
  quiz_id: number;
  question: string;
  options: string[];
  correct_answer: number;
  explanation?: string;
  user_answer?: number;
}

export interface StudyProgress {
  total_flashcards: number;
  reviewed_today: number;
  due_for_review: number;
  mastery_level: number;
}

export interface DashboardStats {
  total_notes: number;
  total_flashcards: number;
  total_quizzes: number;
  recent_sessions: ChatSession[];
  study_progress: StudyProgress;
}

export interface ApiError {
  detail: string;
  message?: string;
}
