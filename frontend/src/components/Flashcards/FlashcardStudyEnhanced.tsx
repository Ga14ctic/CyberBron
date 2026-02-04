import { useState, useEffect } from 'react';
import { CreditCard, Plus, RotateCw, TrendingUp, CheckCircle, XCircle, Sparkles, BookOpen, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { flashcardsService } from '../../services/flashcardsService';
import { notesService } from '../../services/notesService';
import { Flashcard, Note } from '../../types';

export default function FlashcardStudyEnhanced() {
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [dueCards, setDueCards] = useState<Flashcard[]>([]);
  const [currentCard, setCurrentCard] = useState<Flashcard | null>(null);
  const [isFlipped, setIsFlipped] = useState(false);
  const [loading, setLoading] = useState(true);
  const [studyMode, setStudyMode] = useState(false);
  const [newCard, setNewCard] = useState({ front: '', back: '', difficulty: 'medium' as const });
  const [showNewCardForm, setShowNewCardForm] = useState(false);
  const [animating, setAnimating] = useState(false);
  const [notesAvailable, setNotesAvailable] = useState<Note[]>([]);

  useEffect(() => {
    loadFlashcards();
    loadNotes();
  }, []);

  const loadNotes = async () => {
    try {
      const data = await notesService.getNotes();
      setNotesAvailable(data);
    } catch (error) {
      console.error('Failed to load notes:', error);
    }
  };

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
    if (!currentCard || animating) return;

    setAnimating(true);
    
    try {
      await flashcardsService.reviewFlashcard(currentCard.id, difficulty);
      
      // Slide out animation
      setTimeout(() => {
        const remainingDue = dueCards.filter((card) => card.id !== currentCard.id);
        setDueCards(remainingDue);
        
        if (remainingDue.length > 0) {
          setCurrentCard(remainingDue[0]);
          setIsFlipped(false);
        } else {
          setCurrentCard(null);
          setStudyMode(false);
        }
        setAnimating(false);
      }, 300);
    } catch (error) {
      console.error('Failed to review flashcard:', error);
      setAnimating(false);
    }
  };

  const handleCardClick = () => {
    if (!animating) {
      setIsFlipped(!isFlipped);
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
      setIsFlipped(false);
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
        <h1 className="text-3xl font-bold text-cyber-primary flex items-center gap-2">
          <CreditCard className="w-8 h-8" />
          Flashcards
        </h1>
        <button
          onClick={() => setShowNewCardForm(!showNewCardForm)}
          className="btn-secondary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Manual Card</span>
        </button>
      </div>

      {/* Generate from Notes Banner */}
      {notesAvailable.length > 0 && (
        <div className="card bg-gradient-to-r from-cyber-primary/10 to-cyber-secondary/10 border-2 border-cyber-primary/30 mb-6">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-cyber-primary mb-2 flex items-center gap-2">
                <Sparkles className="w-6 h-6" />
                Generate Flashcards from Your Notes
              </h3>
              <p className="text-gray-300 mb-4">
                You have {notesAvailable.length} note{notesAvailable.length !== 1 ? 's' : ''} ready. 
                Generate up to 100 flashcards per note with varied question types and relational understanding.
              </p>
              <div className="flex gap-3">
                <Link to="/" className="btn-primary flex items-center gap-2">
                  <BookOpen className="w-5 h-5" />
                  View Notes
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Total Cards</p>
              <p className="text-3xl font-bold text-cyber-primary">{flashcards.length}</p>
            </div>
            <CreditCard className="w-12 h-12 text-cyber-primary opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Due for Review</p>
              <p className="text-3xl font-bold text-cyber-secondary">{dueCards.length}</p>
            </div>
            <RotateCw className="w-12 h-12 text-cyber-secondary opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
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
        <div className="card mb-8 animate-slide-down">
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
        <div className={`transition-opacity duration-300 ${animating ? 'opacity-0' : 'opacity-100'}`}>
          <div className="text-center mb-4">
            <p className="text-gray-400 text-sm">
              {dueCards.length} card{dueCards.length !== 1 ? 's' : ''} remaining
            </p>
          </div>

          {/* 3D Flip Card */}
          <div className="perspective-1000 mb-6">
            <div 
              className={`relative w-full h-80 transition-transform duration-700 transform-style-3d cursor-pointer ${
                isFlipped ? 'rotate-y-180' : ''
              }`}
              onClick={handleCardClick}
              style={{ transformStyle: 'preserve-3d' }}
            >
              {/* Front Side */}
              <div 
                className={`absolute w-full h-full backface-hidden rounded-lg bg-gradient-to-br from-cyber-gray via-cyber-darker to-cyber-gray border-2 border-cyber-primary shadow-lg shadow-cyber-primary/20 flex items-center justify-center p-8 ${
                  isFlipped ? 'pointer-events-none' : ''
                }`}
                style={{ backfaceVisibility: 'hidden' }}
              >
                <div className="text-center">
                  <p className="text-cyber-secondary text-sm mb-4 font-bold">QUESTION</p>
                  <p className="text-2xl text-gray-100 whitespace-pre-wrap font-medium">
                    {currentCard.front}
                  </p>
                  <p className="text-gray-500 text-sm mt-6 animate-pulse">
                    Click to reveal answer
                  </p>
                </div>
              </div>

              {/* Back Side */}
              <div 
                className={`absolute w-full h-full backface-hidden rounded-lg bg-gradient-to-br from-green-900/30 via-cyber-darker to-green-900/30 border-2 border-cyber-primary shadow-lg shadow-cyber-primary/30 flex items-center justify-center p-8 rotate-y-180 ${
                  !isFlipped ? 'pointer-events-none' : ''
                }`}
                style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
              >
                <div className="text-center">
                  <p className="text-cyber-primary text-sm mb-4 font-bold">ANSWER</p>
                  <p className="text-2xl text-cyber-primary whitespace-pre-wrap font-medium">
                    {currentCard.back}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {isFlipped && (
            <div className="animate-fade-in">
              <p className="text-center text-gray-400 text-sm mb-4">How well did you know this?</p>
              <div className="grid grid-cols-3 gap-4">
                <button
                  onClick={() => handleReview('hard')}
                  disabled={animating}
                  className="bg-red-900/20 border-2 border-red-500 text-red-400 px-6 py-4 rounded-lg font-semibold hover:bg-red-900/40 hover:scale-105 transition-all duration-300 disabled:opacity-50"
                >
                  <XCircle className="w-8 h-8 mx-auto mb-2" />
                  <span className="block text-lg">Hard</span>
                  <span className="block text-xs text-gray-500 mt-1">Review in 1 day</span>
                </button>
                <button
                  onClick={() => handleReview('medium')}
                  disabled={animating}
                  className="bg-yellow-900/20 border-2 border-yellow-500 text-yellow-400 px-6 py-4 rounded-lg font-semibold hover:bg-yellow-900/40 hover:scale-105 transition-all duration-300 disabled:opacity-50"
                >
                  <RotateCw className="w-8 h-8 mx-auto mb-2" />
                  <span className="block text-lg">Medium</span>
                  <span className="block text-xs text-gray-500 mt-1">Review in 3 days</span>
                </button>
                <button
                  onClick={() => handleReview('easy')}
                  disabled={animating}
                  className="bg-green-900/20 border-2 border-green-500 text-green-400 px-6 py-4 rounded-lg font-semibold hover:bg-green-900/40 hover:scale-105 transition-all duration-300 disabled:opacity-50"
                >
                  <CheckCircle className="w-8 h-8 mx-auto mb-2" />
                  <span className="block text-lg">Easy</span>
                  <span className="block text-xs text-gray-500 mt-1">Review in 7 days</span>
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card text-center py-12">
          {dueCards.length > 0 ? (
            <>
              <CreditCard className="w-16 h-16 text-cyber-primary mx-auto mb-4 animate-bounce" />
              <h2 className="text-2xl font-bold text-gray-100 mb-2">
                Ready to Study?
              </h2>
              <p className="text-gray-400 mb-6">
                You have {dueCards.length} card{dueCards.length !== 1 ? 's' : ''} due for review
              </p>
              <button onClick={startStudy} className="btn-primary inline-flex items-center space-x-2 px-8 py-3 text-lg">
                <RotateCw className="w-6 h-6" />
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

      <style>{`
        .perspective-1000 {
          perspective: 1000px;
        }
        
        .transform-style-3d {
          transform-style: preserve-3d;
        }
        
        .backface-hidden {
          backface-visibility: hidden;
        }
        
        .rotate-y-180 {
          transform: rotateY(180deg);
        }
        
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slide-down {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
        
        .animate-slide-down {
          animation: slide-down 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}
