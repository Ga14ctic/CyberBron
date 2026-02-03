import { useState, useEffect } from 'react';
import { Users, Activity, BookOpen, CreditCard, TrendingUp, Shield, CheckCircle, XCircle } from 'lucide-react';

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}

interface Stats {
  total_users: number;
  active_users: number;
  total_notes: number;
  total_flashcards: number;
  total_quizzes: number;
}

export default function AdminDashboard() {
  const [users, setUsers] = useState<User[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      // Mock data for now - replace with actual API calls
      setUsers([
        {
          id: 1,
          username: 'admin',
          email: 'admin@cyberbron.com',
          full_name: 'Admin User',
          is_active: true,
          is_admin: true,
          created_at: '2024-01-01T00:00:00Z',
        },
        {
          id: 2,
          username: 'student1',
          email: 'student1@school.com',
          full_name: 'Student One',
          is_active: true,
          is_admin: false,
          created_at: '2024-01-15T00:00:00Z',
        },
      ]);

      setStats({
        total_users: 25,
        active_users: 23,
        total_notes: 342,
        total_flashcards: 1250,
        total_quizzes: 89,
      });
    } catch (error) {
      console.error('Failed to load admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleUserStatus = async (userId: number, currentStatus: boolean) => {
    try {
      // API call to toggle user status
      setUsers(users.map(u => 
        u.id === userId ? { ...u, is_active: !currentStatus } : u
      ));
    } catch (error) {
      console.error('Failed to toggle user status:', error);
    }
  };

  const toggleAdminStatus = async (userId: number, currentStatus: boolean) => {
    try {
      // API call to toggle admin status
      setUsers(users.map(u => 
        u.id === userId ? { ...u, is_admin: !currentStatus } : u
      ));
    } catch (error) {
      console.error('Failed to toggle admin status:', error);
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
    <div className="max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-cyber-primary flex items-center gap-2">
          <Shield className="w-8 h-8" />
          Admin Dashboard
        </h1>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Total Users</p>
              <p className="text-3xl font-bold text-cyber-primary">{stats?.total_users}</p>
            </div>
            <Users className="w-12 h-12 text-cyber-primary opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Active Users</p>
              <p className="text-3xl font-bold text-green-400">{stats?.active_users}</p>
            </div>
            <Activity className="w-12 h-12 text-green-400 opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Total Notes</p>
              <p className="text-3xl font-bold text-blue-400">{stats?.total_notes}</p>
            </div>
            <BookOpen className="w-12 h-12 text-blue-400 opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Flashcards</p>
              <p className="text-3xl font-bold text-purple-400">{stats?.total_flashcards}</p>
            </div>
            <CreditCard className="w-12 h-12 text-purple-400 opacity-50" />
          </div>
        </div>

        <div className="card hover:scale-105 transition-transform duration-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm mb-1">Quizzes</p>
              <p className="text-3xl font-bold text-yellow-400">{stats?.total_quizzes}</p>
            </div>
            <TrendingUp className="w-12 h-12 text-yellow-400 opacity-50" />
          </div>
        </div>
      </div>

      {/* User Management */}
      <div className="card">
        <h2 className="text-2xl font-bold text-cyber-primary mb-6">User Management</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-cyber-lightgray">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">ID</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Username</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Email</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Full Name</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Status</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Role</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Joined</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr 
                  key={user.id} 
                  className="border-b border-cyber-gray hover:bg-cyber-darkgray transition-colors"
                >
                  <td className="py-3 px-4">{user.id}</td>
                  <td className="py-3 px-4 font-semibold">{user.username}</td>
                  <td className="py-3 px-4 text-gray-400">{user.email}</td>
                  <td className="py-3 px-4">{user.full_name}</td>
                  <td className="py-3 px-4 text-center">
                    {user.is_active ? (
                      <span className="inline-flex items-center gap-1 px-3 py-1 bg-green-900/30 text-green-400 rounded-full text-sm">
                        <CheckCircle className="w-4 h-4" />
                        Active
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-3 py-1 bg-red-900/30 text-red-400 rounded-full text-sm">
                        <XCircle className="w-4 h-4" />
                        Inactive
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {user.is_admin ? (
                      <span className="inline-flex items-center gap-1 px-3 py-1 bg-purple-900/30 text-purple-400 rounded-full text-sm">
                        <Shield className="w-4 h-4" />
                        Admin
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-900/30 text-blue-400 rounded-full text-sm">
                        <Users className="w-4 h-4" />
                        User
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-center text-gray-400 text-sm">
                    {new Date(user.created_at).toLocaleDateString()}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => toggleUserStatus(user.id, user.is_active)}
                        className={`px-3 py-1 rounded text-sm font-semibold transition-all ${
                          user.is_active
                            ? 'bg-red-900/20 text-red-400 hover:bg-red-900/40'
                            : 'bg-green-900/20 text-green-400 hover:bg-green-900/40'
                        }`}
                      >
                        {user.is_active ? 'Deactivate' : 'Activate'}
                      </button>
                      <button
                        onClick={() => toggleAdminStatus(user.id, user.is_admin)}
                        className={`px-3 py-1 rounded text-sm font-semibold transition-all ${
                          user.is_admin
                            ? 'bg-gray-900/20 text-gray-400 hover:bg-gray-900/40'
                            : 'bg-purple-900/20 text-purple-400 hover:bg-purple-900/40'
                        }`}
                        disabled={user.id === 1}
                      >
                        {user.is_admin ? 'Remove Admin' : 'Make Admin'}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
