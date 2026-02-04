import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Search, BookOpen, Trash2, Edit, Tag, Clock, FileText, Zap, TrendingUp, FolderOpen } from 'lucide-react';
import { notesService } from '../../services/notesService';
import { Note } from '../../types';

export default function NotesList() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedFolder, setSelectedFolder] = useState<string>('All');

  useEffect(() => {
    loadNotes();
  }, []);

  const loadNotes = async () => {
    try {
      const data = await notesService.getNotes();
      setNotes(data);
    } catch (error) {
      console.error('Failed to load notes:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteNote = async (noteId: number) => {
    if (!window.confirm('Are you sure you want to delete this note?')) return;
    
    try {
      await notesService.deleteNote(noteId);
      setNotes(notes.filter((n) => n.id !== noteId));
    } catch (error) {
      console.error('Failed to delete note:', error);
    }
  };

  const filteredNotes = notes.filter(
    (note) => {
      const matchesSearch = note.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        note.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        note.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      
      const matchesFolder = selectedFolder === 'All' || note.folder === selectedFolder;
      
      return matchesSearch && matchesFolder;
    }
  );

  // Get unique folders
  const folders = ['All', ...Array.from(new Set(notes.map(n => n.folder)))];

  // Calculate stats
  const totalNotes = notes.length;
  const recentNotes = notes.slice(0, 3);
  const totalWords = notes.reduce((sum, note) => sum + note.content.split(/\s+/).length, 0);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Hero Section */}
      <div className="mb-8">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h1 className="text-4xl font-bold text-cyber-primary mb-2">
              <BookOpen className="inline-block w-10 h-10 mr-3 -mt-1" />
              My Notes
            </h1>
            <p className="text-gray-400">Your knowledge hub - everything starts here</p>
          </div>
          <Link to="/notes/new" className="btn-primary flex items-center space-x-2 text-lg px-6 py-3">
            <Plus className="w-6 h-6" />
            <span>Create Note</span>
          </Link>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="card bg-gradient-to-br from-cyber-gray to-cyber-lightgray">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Total Notes</p>
                <p className="text-3xl font-bold text-cyber-primary">{totalNotes}</p>
              </div>
              <FileText className="w-12 h-12 text-cyber-primary opacity-30" />
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Total Words</p>
                <p className="text-3xl font-bold text-cyber-secondary">{totalWords.toLocaleString()}</p>
              </div>
              <Zap className="w-12 h-12 text-cyber-secondary opacity-30" />
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm mb-1">Folders</p>
                <p className="text-3xl font-bold text-purple-400">{folders.length - 1}</p>
              </div>
              <FolderOpen className="w-12 h-12 text-purple-400 opacity-30" />
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filter Bar */}
      <div className="mb-6 flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search notes by title, content, or tags..."
            className="input-field pl-10 w-full"
            aria-label="Search notes"
          />
        </div>
        <select
          value={selectedFolder}
          onChange={(e) => setSelectedFolder(e.target.value)}
          className="input-field w-48"
          aria-label="Filter by folder"
        >
          {folders.map(folder => (
            <option key={folder} value={folder}>{folder}</option>
          ))}
        </select>
      </div>

      {filteredNotes.length === 0 ? (
        <div className="card text-center py-12">
          <BookOpen className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-400 mb-2">
            {searchQuery ? 'No notes found' : 'No notes yet'}
          </h2>
          <p className="text-gray-500 mb-4">
            {searchQuery
              ? 'Try a different search term'
              : 'Create your first note to get started on your learning journey'}
          </p>
          {!searchQuery && (
            <Link to="/notes/new" className="btn-primary inline-flex items-center space-x-2">
              <Plus className="w-5 h-5" />
              <span>Create Your First Note</span>
            </Link>
          )}
        </div>
      ) : (
        <>
          {/* Notes Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredNotes.map((note) => (
              <div key={note.id} className="card group hover:border-cyber-primary hover:shadow-lg hover:shadow-cyber-primary/20 transition-all">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-gray-100 truncate mb-1">
                      {note.title}
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <span className="flex items-center gap-1">
                        <FolderOpen className="w-3 h-3" />
                        {note.folder}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(note.updated_at || note.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      deleteNote(note.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-300"
                    aria-label="Delete note"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <p className="text-gray-400 text-sm line-clamp-3 mb-4">
                  {note.content.substring(0, 150)}...
                </p>

                {note.tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-4">
                    {note.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center text-xs bg-cyber-lightgray text-cyber-primary px-2 py-1 rounded-full"
                      >
                        <Tag className="w-3 h-3 mr-1" />
                        {tag}
                      </span>
                    ))}
                    {note.tags.length > 3 && (
                      <span className="text-xs text-gray-500">+{note.tags.length - 3}</span>
                    )}
                  </div>
                )}

                <div className="flex gap-2 pt-3 border-t border-cyber-lightgray">
                  <Link
                    to={`/notes/${note.id}`}
                    className="flex-1 btn-secondary text-sm py-2 flex items-center justify-center gap-2"
                  >
                    <Edit className="w-4 h-4" />
                    Edit
                  </Link>
                  <Link
                    to={`/notes/${note.id}`}
                    className="flex-1 btn-primary text-sm py-2 flex items-center justify-center gap-2"
                  >
                    <Zap className="w-4 h-4" />
                    Generate
                  </Link>
                </div>
              </div>
            ))}
          </div>

          {/* Quick Actions Footer */}
          <div className="mt-8 card bg-gradient-to-r from-cyber-gray to-cyber-lightgray">
            <h3 className="text-lg font-semibold text-cyber-primary mb-4">
              What would you like to create from your notes?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-cyber-darker rounded-lg">
                <div className="text-cyber-primary text-3xl mb-2">🎴</div>
                <p className="text-sm text-gray-300">Generate Flashcards</p>
                <p className="text-xs text-gray-500 mt-1">Up to 100 cards per note</p>
              </div>
              <div className="text-center p-4 bg-cyber-darker rounded-lg">
                <div className="text-cyber-secondary text-3xl mb-2">📊</div>
                <p className="text-sm text-gray-300">Create Presentations</p>
                <p className="text-xs text-gray-500 mt-1">Up to 50 slides</p>
              </div>
              <div className="text-center p-4 bg-cyber-darker rounded-lg">
                <div className="text-yellow-400 text-3xl mb-2">📝</div>
                <p className="text-sm text-gray-300">Generate Quizzes</p>
                <p className="text-xs text-gray-500 mt-1">5-20 questions</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
                  {note.title}
                </h3>
                <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Link
                    to={`/notes/${note.id}`}
                    className="text-cyber-primary hover:text-cyber-secondary"
                    aria-label="Edit note"
                  >
                    <Edit className="w-4 h-4" />
                  </Link>
                  <button
                    onClick={() => deleteNote(note.id)}
                    className="text-red-400 hover:text-red-500"
                    aria-label="Delete note"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <p className="text-gray-400 text-sm mb-4 line-clamp-3">
                {note.content.substring(0, 150)}...
              </p>

              {note.tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-3">
                  {note.tags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center space-x-1 text-xs bg-cyber-lightgray text-cyber-primary px-2 py-1 rounded"
                    >
                      <Tag className="w-3 h-3" />
                      <span>{tag}</span>
                    </span>
                  ))}
                </div>
              )}

              <div className="text-xs text-gray-500">
                Updated {new Date(note.updated_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
