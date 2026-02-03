import api from './api';
import { Note } from '../types';

export const notesService = {
  async getNotes(): Promise<Note[]> {
    const response = await api.get<Note[]>('/notes');
    return response.data;
  },

  async getNote(noteId: number): Promise<Note> {
    const response = await api.get<Note>(`/notes/${noteId}`);
    return response.data;
  },

  async createNote(data: Partial<Note>): Promise<Note> {
    const response = await api.post<Note>('/notes', data);
    return response.data;
  },

  async updateNote(noteId: number, data: Partial<Note>): Promise<Note> {
    const response = await api.put<Note>(`/notes/${noteId}`, data);
    return response.data;
  },

  async deleteNote(noteId: number): Promise<void> {
    await api.delete(`/notes/${noteId}`);
  },

  async searchNotes(query: string): Promise<Note[]> {
    const response = await api.get<Note[]>('/notes/search', {
      params: { q: query },
    });
    return response.data;
  },
};
