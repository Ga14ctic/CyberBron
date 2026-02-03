import api from './api';
import { Quiz } from '../types';

export const quizService = {
  async getQuizzes(): Promise<Quiz[]> {
    const response = await api.get<Quiz[]>('/quizzes');
    return response.data;
  },

  async getQuiz(quizId: number): Promise<Quiz> {
    const response = await api.get<Quiz>(`/quizzes/${quizId}`);
    return response.data;
  },

  async createQuiz(data: Partial<Quiz>): Promise<Quiz> {
    const response = await api.post<Quiz>('/quizzes', data);
    return response.data;
  },

  async deleteQuiz(quizId: number): Promise<void> {
    await api.delete(`/quizzes/${quizId}`);
  },

  async submitQuiz(quizId: number, answers: Record<number, number>): Promise<Quiz> {
    const response = await api.post<Quiz>(`/quizzes/${quizId}/submit`, { answers });
    return response.data;
  },

  async generateQuiz(topic: string, numQuestions: number): Promise<Quiz> {
    const response = await api.post<Quiz>('/quizzes/generate', {
      topic,
      num_questions: numQuestions,
    });
    return response.data;
  },
};
