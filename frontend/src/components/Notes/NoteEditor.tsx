import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Save, ArrowLeft, Tag as TagIcon } from 'lucide-react';
import { notesService } from '../../services/notesService';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function NoteEditor() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [preview, setPreview] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingNote, setLoadingNote] = useState(false);

  useEffect(() => {
    if (id && id !== 'new') {
      loadNote(parseInt(id));
    }
  }, [id]);

  const loadNote = async (noteId: number) => {
    setLoadingNote(true);
    try {
      const note = await notesService.getNote(noteId);
      setTitle(note.title);
      setContent(note.content);
      setTags(note.tags);
    } catch (error) {
      console.error('Failed to load note:', error);
    } finally {
      setLoadingNote(false);
    }
  };

  const handleSave = async () => {
    if (!title.trim() || !content.trim()) return;

    setLoading(true);
    try {
      if (id && id !== 'new') {
        await notesService.updateNote(parseInt(id), { title, content, tags });
      } else {
        await notesService.createNote({ title, content, tags });
      }
      navigate('/notes');
    } catch (error) {
      console.error('Failed to save note:', error);
    } finally {
      setLoading(false);
    }
  };

  const addTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()]);
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter((tag) => tag !== tagToRemove));
  };

  const handleTagKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag();
    }
  };

  if (loadingNote) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <Link
          to="/notes"
          className="flex items-center space-x-2 text-gray-400 hover:text-cyber-primary"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Notes</span>
        </Link>
        <button
          onClick={handleSave}
          disabled={loading || !title.trim() || !content.trim()}
          className="btn-primary flex items-center space-x-2"
        >
          <Save className="w-5 h-5" />
          <span>{loading ? 'Saving...' : 'Save Note'}</span>
        </button>
      </div>

      <div className="card mb-4">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Note Title"
          className="input-field text-2xl font-bold mb-4"
          aria-label="Note title"
        />

        <div className="mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <TagIcon className="w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyPress={handleTagKeyPress}
              placeholder="Add tags (press Enter)"
              className="input-field flex-1"
              aria-label="Add tag"
            />
          </div>
          {tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center space-x-1 bg-cyber-lightgray text-cyber-primary px-3 py-1 rounded"
                >
                  <span>{tag}</span>
                  <button
                    onClick={() => removeTag(tag)}
                    className="hover:text-red-400"
                    aria-label={`Remove tag ${tag}`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="mb-4">
          <div className="flex space-x-4 mb-2">
            <button
              onClick={() => setPreview(false)}
              className={`pb-2 ${
                !preview
                  ? 'text-cyber-primary border-b-2 border-cyber-primary'
                  : 'text-gray-400'
              }`}
            >
              Edit
            </button>
            <button
              onClick={() => setPreview(true)}
              className={`pb-2 ${
                preview
                  ? 'text-cyber-primary border-b-2 border-cyber-primary'
                  : 'text-gray-400'
              }`}
            >
              Preview
            </button>
          </div>

          {preview ? (
            <div className="bg-cyber-darker p-4 rounded min-h-[400px]">
              <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert max-w-none">
                {content || '*No content yet*'}
              </ReactMarkdown>
            </div>
          ) : (
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Write your note in Markdown..."
              className="input-field min-h-[400px] font-mono"
              aria-label="Note content"
            />
          )}
        </div>

        <div className="text-sm text-gray-500">
          <p>Supports Markdown formatting</p>
        </div>
      </div>
    </div>
  );
}
