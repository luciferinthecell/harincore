from typing import Dict, Set

class AccessControl:
    def __init__(self):
        self._perm:Dict[str,Set[str]]={
            'admin':{'chat','export','plugin','telemetry'},
            'user':{'chat','plugin'},
            'guest':{'chat'}
        }
        self._roles:Dict[str,str]={}
    def assign_role(self,user,role):
        if role not in self._perm: raise ValueError(role)
        self._roles[user]=role
    def role(self,user): return self._roles.get(user,'guest')
    def allowed(self,user,action): return action in self._perm.get(self.role(user),set())

class DefaultACL:
    """기본 접근 제어 리스트"""
    def __init__(self):
        self.permissions = {
            'admin': {'read', 'write', 'execute', 'delete'},
            'user': {'read', 'write'},
            'guest': {'read'}
        }
    
    def get_permissions(self, role: str) -> Set[str]:
        """역할에 따른 권한 반환"""
        return self.permissions.get(role, set())
    
    def has_permission(self, role: str, permission: str) -> bool:
        """권한 확인"""
        return permission in self.get_permissions(role)
