<script lang="ts">
  export let message: string = '';

  let timeout = 0;
  $: if (message !== '') {
    console.log('message', message);
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      message = '';
    }, 5000);
  }
</script>

{#if message}
  <div class="fixed right-0 top-0 m-4 cursor-pointer" on:click={() => (message = '')}>
    <div class="rounded bg-red-800 p-4 text-white">
      {message}
    </div>
    <div class="bar transition-all duration-500" style="width: 0;"></div>
  </div>
{/if}

<style lang="postcss" scoped>
  .button {
    @apply rounded bg-gray-700 font-normal text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700 dark:disabled:text-black;
  }
</style>
