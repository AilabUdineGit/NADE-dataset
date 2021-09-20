class IntervalMerger(object):
   def merge(self, intervals):
      if len(intervals) == 0:
         return []
      self.quicksort(intervals,0,len(intervals)-1)
      stack = []
      stack.append(intervals[0])
      for i in range(1,len(intervals)):
         last_element= stack[len(stack)-1]
         if last_element[1] >= intervals[i][0]:
            last_element[1] = max(intervals[i][1],last_element[1])
            stack.pop(len(stack)-1)
            stack.append(last_element)
         else:
            stack.append(intervals[i])
      return [tuple(s) for s in stack]

   def partition(self,array,start,end):
      pivot_index = start
      for i in range(start,end):
         if array[i][0]<=array[end][0]:
            array[i],array[pivot_index] =array[pivot_index],array[i]
            pivot_index+=1
      array[end],array[pivot_index] =array[pivot_index],array[end]
      return pivot_index

   def quicksort(self,array,start,end):
      if start<end:
         partition_index = self.partition(array,start,end)
         self.quicksort(array,start,partition_index-1)
         self.quicksort(array, partition_index + 1, end)