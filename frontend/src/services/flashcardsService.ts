import api from './api';
import { Flashcard, StudyProgress } from '../types';

export const flashcardsService = {
  async getFlashcards(): Promise<Flashcard[]> {
    const response = await api.get<Flashcard[]>('/flashcards');
    return response.data;
  },

  async getFlashcard(flashcardId: number): Promise<Flashcard> {
    const response = await api.get<Flashcard>(`/flashcards/${flashcardId}`);
    return response.data;
  },

  async createFlashcard(data: Partial<Flashcard>): Promise<Flashcard> {
    const response = await api.post<Flashcard>('/flashcards', data);
    return response.data;
  },

  async updateFlashcard(flashcardId: number, data: Partial<Flashcard>): Promise<Flashcard> {
    const response = await api.put<Flashcard>(`/flashcards/${flashcardId}`, data);
    return response.data;
  },

  async deleteFlashcard(flashcardId: number): Promise<void> {
    await api.delete(`/flashcards/${flashcardId}`);
  },

  async getDueFlashcards(): Promise<Flashcard[]> {
    const response = await api.get<Flashcard[]>('/flashcards/due');
    return response.data;
  },

  async reviewFlashcard(flashcardId: number, difficulty: 'easy' | 'medium' | 'hard'): Promise<Flashcard> {
    const response = await api.post<Flashcard>(`/flashcards/${flashcardId}/review`, {
      difficulty,
    });
    return response.data;
  },

  async getProgress(): Promise<StudyProgress> {
    const response = await api.get<StudyProgress>('/flashcards/progress');
    return response.data;
  },
};
