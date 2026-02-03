import api from './api';
import { ChatSession, Message } from '../types';

export const chatService = {
  async getSessions(): Promise<ChatSession[]> {
    const response = await api.get<ChatSession[]>('/chat/sessions');
    return response.data;
  },

  async getSession(sessionId: number): Promise<ChatSession> {
    const response = await api.get<ChatSession>(`/chat/sessions/${sessionId}`);
    return response.data;
  },

  async createSession(title?: string): Promise<ChatSession> {
    const response = await api.post<ChatSession>('/chat/sessions', { title });
    return response.data;
  },

  async deleteSession(sessionId: number): Promise<void> {
    await api.delete(`/chat/sessions/${sessionId}`);
  },

  async sendMessage(sessionId: number, content: string): Promise<Message> {
    const response = await api.post<Message>(`/chat/sessions/${sessionId}/messages`, {
      content,
    });
    return response.data;
  },

  async getMessages(sessionId: number): Promise<Message[]> {
    const response = await api.get<Message[]>(`/chat/sessions/${sessionId}/messages`);
    return response.data;
  },
};
