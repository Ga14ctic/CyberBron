import { useState, useEffect } from 'react';
import { ClipboardCheck, Plus, Trophy, CheckCircle, XCircle } from 'lucide-react';
import { quizService } from '../../services/quizService';
import { Quiz } from '../../types';

export default function QuizTake() {
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);
  const [currentQuiz, setCurrentQuiz] = useState<Quiz | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [generateTopic, setGenerateTopic] = useState('');
  const [showGenerateForm, setShowGenerateForm] = useState(false);

  useEffect(() => {
    loadQuizzes();
  }, []);

  const loadQuizzes = async () => {
    try {
      const data = await quizService.getQuizzes();
      setQuizzes(data);
    } catch (error) {
      console.error('Failed to load quizzes:', error);
    } finally {
      setLoading(false);
    }
  };

  const startQuiz = (quiz: Quiz) => {
    setCurrentQuiz(quiz);
    setCurrentQuestionIndex(0);
    setAnswers({});
    setShowResults(false);
  };

  const selectAnswer = (questionId: number, answerIndex: number) => {
    setAnswers({ ...answers, [questionId]: answerIndex });
  };

  const nextQuestion = () => {
    if (currentQuiz && currentQuestionIndex < currentQuiz.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const previousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const submitQuiz = async () => {
    if (!currentQuiz) return;

    try {
      const result = await quizService.submitQuiz(currentQuiz.id, answers);
      setCurrentQuiz(result);
      setShowResults(true);
    } catch (error) {
      console.error('Failed to submit quiz:', error);
    }
  };

  const generateQuiz = async () => {
    if (!generateTopic.trim()) return;

    setGenerating(true);
    try {
      const quiz = await quizService.generateQuiz(generateTopic, 5);
      setQuizzes([quiz, ...quizzes]);
      setGenerateTopic('');
      setShowGenerateForm(false);
      startQuiz(quiz);
    } catch (error) {
      console.error('Failed to generate quiz:', error);
    } finally {
      setGenerating(false);
    }
  };

  const calculateScore = () => {
    if (!currentQuiz) return 0;
    const correct = currentQuiz.questions.filter(
      (q) => answers[q.id] === q.correct_answer
    ).length;
    return Math.round((correct / currentQuiz.questions.length) * 100);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (currentQuiz && !showResults) {
    const currentQuestion = currentQuiz.questions[currentQuestionIndex];
    const progress = ((currentQuestionIndex + 1) / currentQuiz.questions.length) * 100;

    return (
      <div className="max-w-3xl mx-auto">
        <div className="card mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-cyber-primary">{currentQuiz.title}</h2>
            <span className="text-gray-400">
              Question {currentQuestionIndex + 1} of {currentQuiz.questions.length}
            </span>
          </div>

          <div className="w-full bg-cyber-darker rounded-full h-2 mb-6">
            <div
              className="bg-cyber-primary h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="mb-6">
            <h3 className="text-xl text-gray-100 mb-6">{currentQuestion.question}</h3>

            <div className="space-y-3">
              {currentQuestion.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => selectAnswer(currentQuestion.id, index)}
                  className={`w-full text-left p-4 rounded border-2 transition-all ${
                    answers[currentQuestion.id] === index
                      ? 'border-cyber-primary bg-cyber-primary/10'
                      : 'border-cyber-lightgray hover:border-cyber-primary/50'
                  }`}
                >
                  <span className="text-gray-100">{option}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex justify-between">
            <button
              onClick={previousQuestion}
              disabled={currentQuestionIndex === 0}
              className="btn-secondary"
            >
              Previous
            </button>

            {currentQuestionIndex === currentQuiz.questions.length - 1 ? (
              <button
                onClick={submitQuiz}
                disabled={Object.keys(answers).length !== currentQuiz.questions.length}
                className="btn-primary"
              >
                Submit Quiz
              </button>
            ) : (
              <button onClick={nextQuestion} className="btn-primary">
                Next
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (showResults && currentQuiz) {
    const score = calculateScore();

    return (
      <div className="max-w-3xl mx-auto">
        <div className="card text-center mb-6">
          <Trophy className="w-20 h-20 text-cyber-primary mx-auto mb-4" />
          <h2 className="text-3xl font-bold text-cyber-primary mb-2">Quiz Complete!</h2>
          <p className="text-5xl font-bold text-gray-100 mb-4">{score}%</p>
          <p className="text-gray-400 mb-6">
            You got {currentQuiz.questions.filter((q) => answers[q.id] === q.correct_answer).length} out of{' '}
            {currentQuiz.questions.length} correct
          </p>
          <button
            onClick={() => {
              setCurrentQuiz(null);
              setShowResults(false);
            }}
            className="btn-primary"
          >
            Back to Quizzes
          </button>
        </div>

        <div className="card">
          <h3 className="text-xl font-bold text-cyber-primary mb-4">Review Answers</h3>
          <div className="space-y-6">
            {currentQuiz.questions.map((question, index) => {
              const userAnswer = answers[question.id];
              const isCorrect = userAnswer === question.correct_answer;

              return (
                <div key={question.id} className="pb-6 border-b border-cyber-lightgray last:border-0">
                  <div className="flex items-start space-x-2 mb-3">
                    {isCorrect ? (
                      <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
                    )}
                    <h4 className="text-lg text-gray-100">
                      {index + 1}. {question.question}
                    </h4>
                  </div>

                  <div className="ml-8 space-y-2">
                    {question.options.map((option, optIndex) => (
                      <div
                        key={optIndex}
                        className={`p-3 rounded ${
                          optIndex === question.correct_answer
                            ? 'bg-green-900/20 border border-green-500'
                            : optIndex === userAnswer && !isCorrect
                            ? 'bg-red-900/20 border border-red-500'
                            : 'bg-cyber-darker'
                        }`}
                      >
                        <span className="text-gray-100">{option}</span>
                        {optIndex === question.correct_answer && (
                          <span className="text-green-500 ml-2">✓ Correct</span>
                        )}
                        {optIndex === userAnswer && !isCorrect && (
                          <span className="text-red-500 ml-2">✗ Your answer</span>
                        )}
                      </div>
                    ))}

                    {question.explanation && (
                      <div className="mt-3 p-3 bg-cyber-darker rounded">
                        <p className="text-gray-400 text-sm">
                          <strong>Explanation:</strong> {question.explanation}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-cyber-primary">Quizzes</h1>
        <button
          onClick={() => setShowGenerateForm(!showGenerateForm)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Generate Quiz</span>
        </button>
      </div>

      {showGenerateForm && (
        <div className="card mb-8">
          <h2 className="text-xl font-bold text-cyber-primary mb-4">Generate AI Quiz</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Topic</label>
              <input
                type="text"
                value={generateTopic}
                onChange={(e) => setGenerateTopic(e.target.value)}
                placeholder="e.g., SQL Injection, Network Security, Cryptography"
                className="input-field"
                aria-label="Quiz topic"
              />
            </div>
            <div className="flex justify-end space-x-2">
              <button onClick={() => setShowGenerateForm(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={generateQuiz}
                disabled={generating || !generateTopic.trim()}
                className="btn-primary"
              >
                {generating ? 'Generating...' : 'Generate'}
              </button>
            </div>
          </div>
        </div>
      )}

      {quizzes.length === 0 ? (
        <div className="card text-center py-12">
          <ClipboardCheck className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-400 mb-2">No quizzes yet</h2>
          <p className="text-gray-500 mb-4">Generate your first quiz to test your knowledge</p>
          <button
            onClick={() => setShowGenerateForm(true)}
            className="btn-primary inline-flex items-center space-x-2"
          >
            <Plus className="w-5 h-5" />
            <span>Generate Quiz</span>
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {quizzes.map((quiz) => (
            <div key={quiz.id} className="card hover:border-cyber-primary transition-all group">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-gray-100 mb-2 group-hover:text-cyber-primary transition-colors">
                    {quiz.title}
                  </h3>
                  {quiz.description && (
                    <p className="text-gray-400 text-sm mb-3">{quiz.description}</p>
                  )}
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span className="flex items-center space-x-1">
                      <ClipboardCheck className="w-4 h-4" />
                      <span>{quiz.questions.length} questions</span>
                    </span>
                    {quiz.completed_at && quiz.score !== undefined && (
                      <span className="flex items-center space-x-1">
                        <Trophy className="w-4 h-4" />
                        <span>Score: {quiz.score}%</span>
                      </span>
                    )}
                  </div>
                </div>
                <button onClick={() => startQuiz(quiz)} className="btn-primary">
                  {quiz.completed_at ? 'Retake' : 'Start'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
