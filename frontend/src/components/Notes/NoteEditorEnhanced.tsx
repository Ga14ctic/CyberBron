import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Save, ArrowLeft, Maximize2, Minimize2, CreditCard, Presentation, Eye, Edit3, Sparkles, Palette, FileText, Wand2 } from 'lucide-react';
import { notesService } from '../../services/notesService';
import { flashcardsService } from '../../services/flashcardsService';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Note } from '../../types';

type Theme = 'cyber' | 'dark' | 'light' | 'academic';

export default function NoteEditorEnhanced() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [note, setNote] = useState<Note | null>(null);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [folder, setFolder] = useState('General');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [autoSaved, setAutoSaved] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);
  const [splitView, setSplitView] = useState(true); // Default to split view
  const [showGenerateMenu, setShowGenerateMenu] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [theme, setTheme] = useState<Theme>('cyber');
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (id && id !== 'new') {
      loadNote();
    }
  }, [id]);

  // Auto-save functionality
  useEffect(() => {
    if (!title.trim() || !content.trim() || id === 'new') return;

    // Clear existing timer
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }

    // Set new timer for 2 seconds after last change
    autoSaveTimerRef.current = setTimeout(() => {
      autoSaveNote();
    }, 2000);

    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
      }
    };
  }, [title, content, tags, folder]);

  const autoSaveNote = async () => {
    if (!id || id === 'new' || !title.trim() || !content.trim()) return;

    try {
      await notesService.updateNote(parseInt(id), {
        title,
        content,
        tags,
        folder,
      });
      setAutoSaved(true);
      setTimeout(() => setAutoSaved(false), 2000);
    } catch (error) {
      console.error('Auto-save failed:', error);
    }
  };

  const loadNote = async () => {
    if (!id || id === 'new') return;
    
    setLoading(true);
    try {
      const data = await notesService.getNote(parseInt(id));
      setNote(data);
      setTitle(data.title);
      setContent(data.content);
      setTags(data.tags);
      setFolder(data.folder);
    } catch (error) {
      console.error('Failed to load note:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveNote = async () => {
    if (!title.trim() || !content.trim()) return;

    setSaving(true);
    try {
      if (id === 'new') {
        const created = await notesService.createNote({
          title,
          content,
          tags,
          folder,
          source: 'manual',
        });
        navigate(`/notes/${created.id}`);
      } else if (id) {
        await notesService.updateNote(parseInt(id), {
          title,
          content,
          tags,
          folder,
        });
      }
    } catch (error) {
      console.error('Failed to save note:', error);
    } finally {
      setSaving(false);
    }
  };

  const addTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()]);
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter((t) => t !== tag));
  };

  const generateFlashcards = async () => {
    if (!content.trim()) return;
    
    setGenerating(true);
    try {
      await flashcardsService.generateFlashcards({
        content,
        num_cards: 10,
        deck: title || 'Generated from Notes',
      });
      alert('Flashcards generated successfully! Check the Flashcards page.');
      setShowGenerateMenu(false);
    } catch (error) {
      console.error('Failed to generate flashcards:', error);
      alert('Failed to generate flashcards. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const generatePresentation = async () => {
    if (!content.trim()) return;
    
    setGenerating(true);
    try {
      // This would call the presentation API
      alert('Presentation generation coming soon!');
      setShowGenerateMenu(false);
    } catch (error) {
      console.error('Failed to generate presentation:', error);
    } finally {
      setGenerating(false);
    }
  };

  const summarizeNote = async () => {
    if (!content.trim() || !id || id === 'new') return;
    
    setGenerating(true);
    try {
      // Call AI summarization endpoint
      const response = await fetch(`/api/notes/${id}/summarize`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) throw new Error('Summarization failed');
      
      const data = await response.json();
      alert(`Summary:\n\n${data.summary}`);
    } catch (error) {
      console.error('Failed to summarize note:', error);
      alert('Failed to summarize note. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const expandNote = async () => {
    if (!content.trim() || !id || id === 'new') return;
    
    setGenerating(true);
    try {
      // Call AI expansion endpoint
      const response = await fetch(`/api/notes/${id}/expand`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) throw new Error('Expansion failed');
      
      const data = await response.json();
      
      // Offer to replace content with expanded version
      if (confirm('Replace current content with AI-expanded version?')) {
        setContent(data.expanded_content);
      }
    } catch (error) {
      console.error('Failed to expand note:', error);
      alert('Failed to expand note. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    if (!isFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  const containerClass = isFullscreen
    ? 'fixed inset-0 z-50 bg-cyber-darker p-6 overflow-auto'
    : 'max-w-6xl mx-auto';

  const getThemeClasses = () => {
    switch (theme) {
      case 'light':
        return 'bg-white text-gray-900';
      case 'dark':
        return 'bg-gray-900 text-gray-100';
      case 'academic':
        return 'bg-amber-50 text-gray-900';
      case 'cyber':
      default:
        return 'bg-cyber-darker text-gray-100';
    }
  };

  const getEditorThemeClasses = () => {
    switch (theme) {
      case 'light':
        return 'bg-gray-50 text-gray-900 border-gray-300';
      case 'dark':
        return 'bg-gray-800 text-gray-100 border-gray-700';
      case 'academic':
        return 'bg-white text-gray-900 border-amber-300';
      case 'cyber':
      default:
        return 'bg-cyber-gray text-gray-100 border-cyber-lightgray';
    }
  };

  return (
    <div className={`${containerClass} ${getThemeClasses()} transition-colors duration-300`}>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-4">
          {!isFullscreen && (
            <button
              onClick={() => navigate('/notes')}
              className="btn-secondary"
              aria-label="Back to notes"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
          )}
          <h1 className="text-3xl font-bold text-cyber-primary flex items-center gap-2">
            <Edit3 className="w-8 h-8" />
            {id === 'new' ? 'New Note' : 'Edit Note'}
          </h1>
          {autoSaved && (
            <span className="text-sm text-cyber-primary animate-pulse">
              ✓ Auto-saved
            </span>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Theme Selector */}
          <div className="relative">
            <button
              onClick={() => setShowThemeMenu(!showThemeMenu)}
              className="btn-secondary"
              title="Change theme"
            >
              <Palette className="w-5 h-5" />
            </button>
            
            {showThemeMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-cyber-gray border border-cyber-lightgray rounded-lg shadow-lg z-10">
                {(['cyber', 'dark', 'light', 'academic'] as Theme[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => {
                      setTheme(t);
                      setShowThemeMenu(false);
                    }}
                    className={`w-full px-4 py-2 text-left hover:bg-cyber-lightgray transition-colors ${
                      theme === t ? 'bg-cyber-lightgray text-cyber-primary' : ''
                    }`}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)} Theme
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={() => setSplitView(!splitView)}
            className="btn-secondary"
            title="Toggle split view"
          >
            <Eye className="w-5 h-5" />
          </button>
          
          <button
            onClick={toggleFullscreen}
            className="btn-secondary"
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>
          
          <div className="relative">
            <button
              onClick={() => setShowGenerateMenu(!showGenerateMenu)}
              className="btn-secondary flex items-center space-x-2"
              disabled={!content.trim()}
            >
              <Sparkles className="w-5 h-5" />
              <span>AI Tools</span>
            </button>
            
            {showGenerateMenu && (
              <div className="absolute right-0 mt-2 w-64 bg-cyber-gray border border-cyber-lightgray rounded-lg shadow-lg z-10">
                <button
                  onClick={generateFlashcards}
                  disabled={generating}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <CreditCard className="w-5 h-5 text-cyber-primary" />
                  <span>Generate Flashcards</span>
                </button>
                <button
                  onClick={generatePresentation}
                  disabled={generating}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <Presentation className="w-5 h-5 text-cyber-secondary" />
                  <span>Generate Presentation</span>
                </button>
                <hr className="border-cyber-lightgray" />
                <button
                  onClick={summarizeNote}
                  disabled={generating || id === 'new'}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <FileText className="w-5 h-5 text-blue-400" />
                  <span>AI Summarize</span>
                </button>
                <button
                  onClick={expandNote}
                  disabled={generating || id === 'new'}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <Wand2 className="w-5 h-5 text-purple-400" />
                  <span>AI Expand Content</span>
                </button>
              </div>
            )}
          </div>
          
          <button
            onClick={saveNote}
            disabled={saving || !title.trim() || !content.trim()}
            className="btn-primary flex items-center space-x-2"
          >
            <Save className="w-5 h-5" />
            <span>{saving ? 'Saving...' : 'Save'}</span>
          </button>
        </div>
      </div>

      {/* Note Metadata */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium mb-2">Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="input-field text-xl font-bold"
            placeholder="Enter note title..."
            aria-label="Note title"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">Folder</label>
          <select
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            className="input-field"
            aria-label="Note folder"
          >
            <option value="General">General</option>
            <option value="Unit 1">Unit 1</option>
            <option value="Unit 2">Unit 2</option>
            <option value="Unit 3">Unit 3</option>
            <option value="Unit 4">Unit 4</option>
            <option value="Projects">Projects</option>
            <option value="Resources">Resources</option>
          </select>
        </div>
      </div>

      {/* Tags */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Tags</label>
        <div className="flex flex-wrap gap-2 mb-2">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center space-x-1 text-sm bg-cyber-lightgray text-cyber-primary px-3 py-1 rounded-full"
            >
              <span>{tag}</span>
              <button
                onClick={() => removeTag(tag)}
                className="hover:text-cyber-secondary"
                aria-label={`Remove ${tag} tag`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="flex space-x-2">
          <input
            type="text"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && addTag()}
            className="input-field"
            placeholder="Add a tag..."
            aria-label="Add tag"
          />
          <button onClick={addTag} className="btn-secondary">
            Add
          </button>
        </div>
      </div>

      {/* Editor/Preview */}
      <div className={`grid ${splitView ? 'grid-cols-2 gap-4' : 'grid-cols-1'}`}>
        {/* Editor */}
        {(!previewMode || splitView) && (
          <div className="card">
            <label className="block text-sm font-medium mb-2">Content (Markdown Supported)</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className={`w-full px-4 py-3 rounded-lg border-2 font-mono text-base leading-relaxed focus:ring-2 focus:ring-cyber-primary focus:border-cyber-primary transition-all ${getEditorThemeClasses()}`}
              rows={isFullscreen ? 30 : 20}
              placeholder="Write your notes here... Markdown is supported!"
              aria-label="Note content"
            />
          </div>
        )}

        {/* Preview */}
        {(previewMode || splitView) && (
          <div className="card">
            <label className="block text-sm font-medium mb-2">Preview</label>
            <div className={`prose prose-lg max-w-none p-6 rounded-lg min-h-[500px] ${getEditorThemeClasses()}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content || '*No content to preview*'}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="mt-6 text-center text-sm text-gray-500">
        <p>
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">Ctrl+S</kbd> to save |{' '}
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">F11</kbd> for fullscreen |{' '}
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">Ctrl+E</kbd> toggle split view |{' '}
          <span className="text-cyber-primary">Auto-save enabled</span>
        </p>
      </div>
    </div>
  );
}
