import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Save, ArrowLeft, Maximize2, Minimize2, CreditCard, Presentation, Eye, Edit3, Sparkles } from 'lucide-react';
import { notesService } from '../../services/notesService';
import { flashcardsService } from '../../services/flashcardsService';
import ReactMarkdown from 'react-markdown';
import { Note } from '../../types';

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
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);
  const [splitView, setSplitView] = useState(false);
  const [showGenerateMenu, setShowGenerateMenu] = useState(false);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    if (id && id !== 'new') {
      loadNote();
    }
  }, [id]);

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

  return (
    <div className={containerClass}>
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
        </div>
        
        <div className="flex items-center space-x-2">
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
              <span>Generate</span>
            </button>
            
            {showGenerateMenu && (
              <div className="absolute right-0 mt-2 w-56 bg-cyber-gray border border-cyber-lightgray rounded-lg shadow-lg z-10">
                <button
                  onClick={generateFlashcards}
                  disabled={generating}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <CreditCard className="w-5 h-5 text-cyber-primary" />
                  <span>Create Flashcards</span>
                </button>
                <button
                  onClick={generatePresentation}
                  disabled={generating}
                  className="w-full px-4 py-3 text-left hover:bg-cyber-lightgray transition-colors flex items-center space-x-2"
                >
                  <Presentation className="w-5 h-5 text-cyber-secondary" />
                  <span>Create Presentation</span>
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
              className="input-field font-mono"
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
            <div className="markdown-content bg-cyber-darker p-6 rounded-lg min-h-[500px]">
              <ReactMarkdown>{content || '*No content to preview*'}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="mt-6 text-center text-sm text-gray-500">
        <p>
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">Ctrl+S</kbd> to save |{' '}
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">F11</kbd> for fullscreen |{' '}
          <kbd className="px-2 py-1 bg-cyber-gray rounded text-xs">Ctrl+P</kbd> for preview
        </p>
      </div>
    </div>
  );
}
