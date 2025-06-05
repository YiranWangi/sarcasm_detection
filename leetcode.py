
class Solution:
    def permute(self,nums):
        path = [] # 记录当前已经选择的排列元素
        result = [] # 来存放所有的最终排列结果
        used = [False]*len(nums)

        # 结束条件：nums长度等于len(path)
        n = len(nums)

        def dfs(path, nums, used):
            if len(path) == n:
                result.append(path[:])
                return

            for i in range(len(nums)):
                if used[i] == False:
                    path.append(nums[i])
                    used[i] == True
                    dfs(path, nums, used)
                    path.pop()
                    used[i] = False

        dfs(path,nums,used)

        return result








