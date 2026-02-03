import { useState, useEffect } from 'react';
import { CreditCard, Plus, RotateCw, TrendingUp, CheckCircle, XCircle } from 'lucide-react';
import { flashcardsService } from '../../services/flashcardsService';
import { Flashcard } from '../../types';

export default function FlashcardStudy() {
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [dueCards, setDueCards] = useState<Flashcard[]>([]);
  const [currentCard, setCurrentCard] = useState<Flashcard | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [loading, setLoading] = useState(true);
  const [studyMode, setStudyMode] = useState(false);
  const [newCard, setNewCard] = useState({ front: '', back: '', difficulty: 'medium' as const });
  const [showNewCardForm, setShowNewCardForm] = useState(false);

  useEffect(() => {
    loadFlashcards();
  }, []);

  const loadFlashcards = async () => {
    try {
      const [allCards, due] = await Promise.all([
        flashcardsService.getFlashcards(),
        flashcardsService.getDueFlashcards(),
      ]);
      setFlashcards(allCards);
      setDueCards(due);
      if (due.length > 0) {
        setCurrentCard(due[0]);
      }
    } catch (error) {
      console.error('Failed to load flashcards:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReview = async (difficulty: 'easy' | 'medium' | 'hard') => {
    if (!currentCard) return;

    try {
      await flashcardsService.reviewFlashcard(currentCard.id, difficulty);
      const remainingDue = dueCards.filter((card) => card.id !== currentCard.id);
      setDueCards(remainingDue);
      
      if (remainingDue.length > 0) {
        setCurrentCard(remainingDue[0]);
        setShowAnswer(false);
      } else {
        setCurrentCard(null);
        setStudyMode(false);
      }
    } catch (error) {
      console.error('Failed to review flashcard:', error);
    }
  };

  const createFlashcard = async () => {
    if (!newCard.front.trim() || !newCard.back.trim()) return;

    try {
      const created = await flashcardsService.createFlashcard(newCard);
      setFlashcards([...flashcards, created]);
      setNewCard({ front: '', back: '', difficulty: 'medium' });
      setShowNewCardForm(false);
    } catch (error) {
      console.error('Failed to create flashcard:', error);
    }
  };

  const startStudy = () => {
    if (dueCards.length > 0) {
      setCurrentCard(dueCards[0]);
      setStudyMode(true);
      setShowAnswer(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-cyber-primary">Flashcards</h1>
        <button
          onClick={() => setShowNewCardForm(!showNewCardForm)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>New Flashcard</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Total Cards</p>
              <p className="text-3xl font-bold text-cyber-primary">{flashcards.length}</p>
            </div>
            <CreditCard className="w-12 h-12 text-cyber-primary opacity-50" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Due for Review</p>
              <p className="text-3xl font-bold text-cyber-secondary">{dueCards.length}</p>
            </div>
            <RotateCw className="w-12 h-12 text-cyber-secondary opacity-50" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Mastery</p>
              <p className="text-3xl font-bold text-purple-400">
                {flashcards.length > 0 ? Math.round((flashcards.length - dueCards.length) / flashcards.length * 100) : 0}%
              </p>
            </div>
            <TrendingUp className="w-12 h-12 text-purple-400 opacity-50" />
          </div>
        </div>
      </div>

      {showNewCardForm && (
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-cyber-primary mb-4">Create New Flashcard</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Front (Question)</label>
              <textarea
                value={newCard.front}
                onChange={(e) => setNewCard({ ...newCard, front: e.target.value })}
                className="input-field"
                rows={3}
                placeholder="Enter the question or term..."
                aria-label="Flashcard front"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Back (Answer)</label>
              <textarea
                value={newCard.back}
                onChange={(e) => setNewCard({ ...newCard, back: e.target.value })}
                className="input-field"
                rows={3}
                placeholder="Enter the answer or definition..."
                aria-label="Flashcard back"
              />
            </div>
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowNewCardForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={createFlashcard}
                disabled={!newCard.front.trim() || !newCard.back.trim()}
                className="btn-primary"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {studyMode && currentCard ? (
        <div className="card">
          <div className="text-center mb-4">
            <p className="text-gray-400 text-sm">
              {dueCards.length} card{dueCards.length !== 1 ? 's' : ''} remaining
            </p>
          </div>

          <div
            className="bg-cyber-darker p-8 rounded-lg min-h-[300px] flex items-center justify-center cursor-pointer mb-6"
            onClick={() => setShowAnswer(!showAnswer)}
          >
            <div className="text-center">
              {!showAnswer ? (
                <>
                  <p className="text-gray-400 text-sm mb-4">Question</p>
                  <p className="text-2xl text-gray-100 whitespace-pre-wrap">
                    {currentCard.front}
                  </p>
                  <p className="text-gray-500 text-sm mt-4">Click to reveal answer</p>
                </>
              ) : (
                <>
                  <p className="text-gray-400 text-sm mb-4">Answer</p>
                  <p className="text-2xl text-cyber-primary whitespace-pre-wrap">
                    {currentCard.back}
                  </p>
                </>
              )}
            </div>
          </div>

          {showAnswer && (
            <div>
              <p className="text-center text-gray-400 text-sm mb-4">How well did you know this?</p>
              <div className="grid grid-cols-3 gap-4">
                <button
                  onClick={() => handleReview('hard')}
                  className="bg-red-900/20 border border-red-500 text-red-400 px-4 py-3 rounded font-semibold hover:bg-red-900/30 transition-all"
                >
                  <XCircle className="w-6 h-6 mx-auto mb-1" />
                  Hard
                </button>
                <button
                  onClick={() => handleReview('medium')}
                  className="bg-yellow-900/20 border border-yellow-500 text-yellow-400 px-4 py-3 rounded font-semibold hover:bg-yellow-900/30 transition-all"
                >
                  <RotateCw className="w-6 h-6 mx-auto mb-1" />
                  Medium
                </button>
                <button
                  onClick={() => handleReview('easy')}
                  className="bg-green-900/20 border border-green-500 text-green-400 px-4 py-3 rounded font-semibold hover:bg-green-900/30 transition-all"
                >
                  <CheckCircle className="w-6 h-6 mx-auto mb-1" />
                  Easy
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card text-center py-12">
          {dueCards.length > 0 ? (
            <>
              <CreditCard className="w-16 h-16 text-cyber-primary mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-100 mb-2">
                Ready to Study?
              </h2>
              <p className="text-gray-400 mb-6">
                You have {dueCards.length} card{dueCards.length !== 1 ? 's' : ''} due for review
              </p>
              <button onClick={startStudy} className="btn-primary inline-flex items-center space-x-2">
                <RotateCw className="w-5 h-5" />
                <span>Start Studying</span>
              </button>
            </>
          ) : (
            <>
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-100 mb-2">
                All Caught Up!
              </h2>
              <p className="text-gray-400 mb-6">
                No cards are due for review right now. Great job!
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
