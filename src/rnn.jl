





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://assets-cdn.github.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/frameworks-c9193575f18b28be82c0a963e144ff6fa7a809dd8ae003a1d1e5979bed3f7f00.css" media="all" rel="stylesheet" />
  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/github-a3cc9068f5810193ef5f7d4b78da3455aeb3040017a69a87038a95d81fa2a583.css" media="all" rel="stylesheet" />
  
  
  <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/site-44b6bba3881278f33c221b6526379b55fbd098af3e553f54e81cab4c9a517c8e.css" media="all" rel="stylesheet" />
  

  <meta name="viewport" content="width=device-width">
  
  <title>Knet.jl/rnn.jl at master · denizyuret/Knet.jl · GitHub</title>
  <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    
    <meta content="https://avatars2.githubusercontent.com/u/1822118?s=400&amp;v=4" property="og:image" /><meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="denizyuret/Knet.jl" property="og:title" /><meta content="https://github.com/denizyuret/Knet.jl" property="og:url" /><meta content="Knet.jl - Koç University deep learning framework." property="og:description" />

  <link rel="assets" href="https://assets-cdn.github.com/">
  
  <meta name="pjax-timeout" content="1000">
  
  <meta name="request-id" content="51A1:4230:369AB4E:5E4C2F3:5A2C3BF5" data-pjax-transient>
  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

  <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
<meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
<meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">
    <meta name="google-analytics" content="UA-3769691-2">

<meta content="collector.githubapp.com" name="octolytics-host" /><meta content="github" name="octolytics-app-id" /><meta content="https://collector.githubapp.com/github-external/browser_event" name="octolytics-event-url" /><meta content="51A1:4230:369AB4E:5E4C2F3:5A2C3BF5" name="octolytics-dimension-request_id" /><meta content="iad" name="octolytics-dimension-region_edge" /><meta content="iad" name="octolytics-dimension-region_render" />
<meta content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" name="analytics-location" />




  <meta class="js-ga-set" name="dimension1" content="Logged Out">


  

      <meta name="hostname" content="github.com">
  <meta name="user-login" content="">

      <meta name="expected-hostname" content="github.com">
    <meta name="js-proxy-site-detection-payload" content="YzQwZjgyMzk3MWNhOWM5NjZkMzk3NjViMGNkYzRjMDEwMzAyYWFkMzcyYWQxNjU3YjkzZTRhZjU5ZDVjOTAzMXx7InJlbW90ZV9hZGRyZXNzIjoiMjEyLjE3NS4zMi4xMzciLCJyZXF1ZXN0X2lkIjoiNTFBMTo0MjMwOjM2OUFCNEU6NUU0QzJGMzo1QTJDM0JGNSIsInRpbWVzdGFtcCI6MTUxMjg0ODM3MywiaG9zdCI6ImdpdGh1Yi5jb20ifQ==">


  <meta name="html-safe-nonce" content="c5f8c40f2bbba2a783d6bf483323659fc67a7a38">

  <meta http-equiv="x-pjax-version" content="e27877525cd6a2549bd009e08999c832">
  

      <link href="https://github.com/denizyuret/Knet.jl/commits/master.atom" rel="alternate" title="Recent Commits to Knet.jl:master" type="application/atom+xml">

  <meta name="description" content="Knet.jl - Koç University deep learning framework.">
  <meta name="go-import" content="github.com/denizyuret/Knet.jl git https://github.com/denizyuret/Knet.jl.git">

  <meta content="1822118" name="octolytics-dimension-user_id" /><meta content="denizyuret" name="octolytics-dimension-user_login" /><meta content="43400734" name="octolytics-dimension-repository_id" /><meta content="denizyuret/Knet.jl" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="43400734" name="octolytics-dimension-repository_network_root_id" /><meta content="denizyuret/Knet.jl" name="octolytics-dimension-repository_network_root_nwo" /><meta content="false" name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" />


    <link rel="canonical" href="https://github.com/denizyuret/Knet.jl/blob/master/src/rnn.jl" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://assets-cdn.github.com/favicon.ico">

<meta name="theme-color" content="#1e2327">



  </head>

  <body class="logged-out env-production page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="px-2 py-4 show-on-focus js-skip-to-content">Skip to content</a>
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>

    
    
    
      



        <header class="Header header-logged-out  position-relative f4 py-3" role="banner">
  <div class="container-lg d-flex px-3">
    <div class="d-flex flex-justify-between flex-items-center">
      <a class="header-logo-invertocat my-0" href="https://github.com/" aria-label="Homepage" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
        <svg aria-hidden="true" class="octicon octicon-mark-github" height="32" version="1.1" viewBox="0 0 16 16" width="32"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
      </a>

    </div>

    <div class="HeaderMenu HeaderMenu--bright d-flex flex-justify-between flex-auto">
        <nav class="mt-0">
          <ul class="d-flex list-style-none">
              <li class="ml-2">
                <a href="/features" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:features" data-selected-links="/features /features/project-management /features/code-review /features/project-management /features/integrations /features">
                  Features
</a>              </li>
              <li class="ml-4">
                <a href="/business" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:business" data-selected-links="/business /business/security /business/customers /business">
                  Business
</a>              </li>

              <li class="ml-4">
                <a href="/explore" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore">
                  Explore
</a>              </li>

              <li class="ml-4">
                    <a href="/marketplace" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:marketplace" data-selected-links=" /marketplace">
                      Marketplace
</a>              </li>
              <li class="ml-4">
                <a href="/pricing" class="js-selected-navigation-item HeaderNavlink px-0 py-2 m-0" data-ga-click="Header, click, Nav menu - item:pricing" data-selected-links="/pricing /pricing/developer /pricing/team /pricing/business-hosted /pricing/business-enterprise /pricing">
                  Pricing
</a>              </li>
          </ul>
        </nav>

      <div class="d-flex">
          <div class="d-lg-flex flex-items-center mr-3">
            <div class="header-search scoped-search site-scoped-search js-site-search" role="search">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/denizyuret/Knet.jl/search" class="js-site-search-form" data-scoped-search-url="/denizyuret/Knet.jl/search" data-unscoped-search-url="/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <label class="form-control header-search-wrapper js-chromeless-input-container">
        <a href="/denizyuret/Knet.jl/blob/master/src/rnn.jl" class="header-search-scope no-underline">This repository</a>
      <input type="text"
        class="form-control header-search-input js-site-search-focus js-site-search-field is-clearable"
        data-hotkey="s"
        name="q"
        value=""
        placeholder="Search"
        aria-label="Search this repository"
        data-unscoped-placeholder="Search GitHub"
        data-scoped-placeholder="Search"
        autocapitalize="off">
        <input type="hidden" class="js-site-search-type-field" name="type" >
    </label>
</form></div>

          </div>

        <span class="d-inline-block">
            <div class="HeaderNavlink px-0 py-2 m-0">
              <a class="text-bold text-white no-underline" href="/login?return_to=%2Fdenizyuret%2FKnet.jl%2Fblob%2Fmaster%2Fsrc%2Frnn.jl" data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">Sign in</a>
                <span class="text-gray">or</span>
                <a class="text-bold text-white no-underline" href="/join?source=header-repo" data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">Sign up</a>
            </div>
        </span>
      </div>
    </div>
  </div>
</header>


  </div>

  <div id="start-of-content" class="show-on-focus"></div>

    <div id="js-flash-container">
</div>



  <div role="main">
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode">
    <div id="js-repo-pjax-container" data-pjax-container>
      





  <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav ">
    <div class="repohead-details-container clearfix container ">

      <ul class="pagehead-actions">
  <li>
      <a href="/login?return_to=%2Fdenizyuret%2FKnet.jl"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to watch a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
    Watch
  </a>
  <a class="social-count" href="/denizyuret/Knet.jl/watchers"
     aria-label="58 users are watching this repository">
    58
  </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fdenizyuret%2FKnet.jl"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to star a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-star" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74z"/></svg>
    Star
  </a>

    <a class="social-count js-social-count" href="/denizyuret/Knet.jl/stargazers"
      aria-label="409 users starred this repository">
      409
    </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fdenizyuret%2FKnet.jl"
        class="btn btn-sm btn-with-count tooltipped tooltipped-n"
        aria-label="You must be signed in to fork a repository" rel="nofollow">
        <svg aria-hidden="true" class="octicon octicon-repo-forked" height="16" version="1.1" viewBox="0 0 10 16" width="10"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
        Fork
      </a>

    <a href="/denizyuret/Knet.jl/network" class="social-count"
       aria-label="97 users forked this repository">
      97
    </a>
  </li>
</ul>

      <h1 class="public ">
  <svg aria-hidden="true" class="octicon octicon-repo" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author" itemprop="author"><a href="/denizyuret" class="url fn" rel="author">denizyuret</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a href="/denizyuret/Knet.jl" data-pjax="#js-repo-pjax-container">Knet.jl</a></strong>

</h1>

    </div>
    
<nav class="reponav js-repo-nav js-sidenav-container-pjax container"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
     role="navigation"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/denizyuret/Knet.jl" class="js-selected-navigation-item selected reponav-item" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /denizyuret/Knet.jl" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-code" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a href="/denizyuret/Knet.jl/issues" class="js-selected-navigation-item reponav-item" data-hotkey="g i" data-selected-links="repo_issues repo_labels repo_milestones /denizyuret/Knet.jl/issues" itemprop="url">
        <svg aria-hidden="true" class="octicon octicon-issue-opened" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg>
        <span itemprop="name">Issues</span>
        <span class="Counter">51</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/denizyuret/Knet.jl/pulls" class="js-selected-navigation-item reponav-item" data-hotkey="g p" data-selected-links="repo_pulls /denizyuret/Knet.jl/pulls" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-git-pull-request" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">5</span>
      <meta itemprop="position" content="3">
</a>  </span>

    <a href="/denizyuret/Knet.jl/projects" class="js-selected-navigation-item reponav-item" data-hotkey="g b" data-selected-links="repo_projects new_repo_project repo_project /denizyuret/Knet.jl/projects">
      <svg aria-hidden="true" class="octicon octicon-project" height="16" version="1.1" viewBox="0 0 15 16" width="15"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      Projects
      <span class="Counter" >0</span>
</a>


  <a href="/denizyuret/Knet.jl/pulse" class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph pulse /denizyuret/Knet.jl/pulse">
    <svg aria-hidden="true" class="octicon octicon-graph" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
    Insights
</a>

</nav>


  </div>

<div class="container new-discussion-timeline experiment-repo-nav">
  <div class="repository-content">

    
  <a href="/denizyuret/Knet.jl/blob/56d3fbc2799ddb022df688418639da6b85be9a35/src/rnn.jl" class="d-none js-permalink-shortcut" data-hotkey="y">Permalink</a>

  <!-- blob contrib key: blob_contributors:v21:668729ce2b533d8cbf3991acfaee8c25 -->

  <div class="file-navigation js-zeroclipboard-container">
    
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class=" btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    
    type="button" aria-label="Switch branches or tags" aria-expanded="false" aria-haspopup="true">
      <i>Branch:</i>
      <span class="js-select-button css-truncate-target">master</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax>

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg aria-label="Close" class="octicon octicon-x js-menu-close" height="16" role="img" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/CuArrays/src/rnn.jl"
               data-name="CuArrays"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                CuArrays
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/GPUArrays/src/rnn.jl"
               data-name="GPUArrays"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                GPUArrays
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/conv/src/rnn.jl"
               data-name="conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/conv5/src/rnn.jl"
               data-name="conv5"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                conv5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/cpuconv/src/rnn.jl"
               data-name="cpuconv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cpuconv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/dev/src/rnn.jl"
               data-name="dev"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dev
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/droptest/src/rnn.jl"
               data-name="droptest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                droptest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/gh-pages/src/rnn.jl"
               data-name="gh-pages"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gh-pages
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/gpuprof/src/rnn.jl"
               data-name="gpuprof"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gpuprof
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/handout-mtg/src/rnn.jl"
               data-name="handout-mtg"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                handout-mtg
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/ilkarman/src/rnn.jl"
               data-name="ilkarman"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ilkarman
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/ipa/src/rnn.jl"
               data-name="ipa"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ipa
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/itembased/src/rnn.jl"
               data-name="itembased"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                itembased
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/lcn/src/rnn.jl"
               data-name="lcn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lcn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/denizyuret/Knet.jl/blob/master/src/rnn.jl"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/mnist4d/src/rnn.jl"
               data-name="mnist4d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mnist4d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/mochaconv/src/rnn.jl"
               data-name="mochaconv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mochaconv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/naacl16/src/rnn.jl"
               data-name="naacl16"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                naacl16
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/nce/src/rnn.jl"
               data-name="nce"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nce
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/nce8/src/rnn.jl"
               data-name="nce8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nce8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/newdocs/src/rnn.jl"
               data-name="newdocs"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                newdocs
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/newupdate/src/rnn.jl"
               data-name="newupdate"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                newupdate
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/perceptron/src/rnn.jl"
               data-name="perceptron"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perceptron
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/quaertym-patch-1/src/rnn.jl"
               data-name="quaertym-patch-1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                quaertym-patch-1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0/src/rnn.jl"
               data-name="release-0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.0/src/rnn.jl"
               data-name="release-0.0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.1/src/rnn.jl"
               data-name="release-0.1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.2/src/rnn.jl"
               data-name="release-0.2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.2.3/src/rnn.jl"
               data-name="release-0.2.3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.2.3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.3/src/rnn.jl"
               data-name="release-0.3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.4/src/rnn.jl"
               data-name="release-0.4"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.4
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.5/src/rnn.jl"
               data-name="release-0.5"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.6/src/rnn.jl"
               data-name="release-0.6"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.6
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/release-0.7/src/rnn.jl"
               data-name="release-0.7"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                release-0.7
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/rnn/src/rnn.jl"
               data-name="rnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                rnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/rnncpu/src/rnn.jl"
               data-name="rnncpu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                rnncpu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/rtfd2github/src/rnn.jl"
               data-name="rtfd2github"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                rtfd2github
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/tentencopy/src/rnn.jl"
               data-name="tentencopy"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tentencopy
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/transpose-0.7/src/rnn.jl"
               data-name="transpose-0.7"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                transpose-0.7
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/denizyuret/Knet.jl/blob/windows/src/rnn.jl"
               data-name="windows"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                windows
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.5/src/rnn.jl"
              data-name="v0.8.5"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.5">
                v0.8.5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.4/src/rnn.jl"
              data-name="v0.8.4"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.4">
                v0.8.4
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.3/src/rnn.jl"
              data-name="v0.8.3"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.3">
                v0.8.3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.2/src/rnn.jl"
              data-name="v0.8.2"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.2">
                v0.8.2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.1/src/rnn.jl"
              data-name="v0.8.1"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.1">
                v0.8.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.8.0/src/rnn.jl"
              data-name="v0.8.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.8.0">
                v0.8.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/denizyuret/Knet.jl/tree/v0.7.3/src/rnn.jl"
              data-name="v0.7.3"
              data-skip-pjax="true"
              rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="v0.7.3">
                v0.7.3
              </span>
            </a>
        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

    <div class="BtnGroup float-right">
      <a href="/denizyuret/Knet.jl/find/master"
            class="js-pjax-capture-input btn btn-sm BtnGroup-item"
            data-pjax
            data-hotkey="t">
        Find file
      </a>
      <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm BtnGroup-item tooltipped tooltipped-s" data-copied-hint="Copied!" type="button">Copy path</button>
    </div>
    <div class="breadcrumb js-zeroclipboard-target">
      <span class="repo-root js-repo-root"><span class="js-path-segment"><a href="/denizyuret/Knet.jl"><span>Knet.jl</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a href="/denizyuret/Knet.jl/tree/master/src"><span>src</span></a></span><span class="separator">/</span><strong class="final-path">rnn.jl</strong>
    </div>
  </div>


  
  <div class="commit-tease">
      <span class="float-right">
        <a class="commit-tease-sha" href="/denizyuret/Knet.jl/commit/77a6c3f31a84302ab06c3c297d49ed4b2bf9d98c" data-pjax>
          77a6c3f
        </a>
        <relative-time datetime="2017-12-03T17:41:57Z">Dec 3, 2017</relative-time>
      </span>
      <div>
        <img alt="@denizyuret" class="avatar" height="20" src="https://avatars3.githubusercontent.com/u/1822118?s=40&amp;v=4" width="20" />
        <a href="/denizyuret" class="user-mention" rel="author">denizyuret</a>
          <a href="/denizyuret/Knet.jl/commit/77a6c3f31a84302ab06c3c297d49ed4b2bf9d98c" class="message" data-pjax="true" title="CPU support for rnnparam, rnnparams, rnninit, rnnforw.
(Except dropout, bidirectional, and batchSizes)">CPU support for rnnparam, rnnparams, rnninit, rnnforw.</a>
      </div>

    <div class="commit-tease-contributors">
      <button type="button" class="btn-link muted-link contributors-toggle" data-facebox="#blob_contributors_box">
        <strong>1</strong>
         contributor
      </button>
      
    </div>

    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header" data-facebox-id="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list" data-facebox-id="facebox-description">
          <li class="facebox-user-list-item">
            <img alt="@denizyuret" height="24" src="https://avatars2.githubusercontent.com/u/1822118?s=48&amp;v=4" width="24" />
            <a href="/denizyuret">denizyuret</a>
          </li>
      </ul>
    </div>
  </div>


  <div class="file">
    <div class="file-header">
  <div class="file-actions">

    <div class="BtnGroup">
      <a href="/denizyuret/Knet.jl/raw/master/src/rnn.jl" class="btn btn-sm BtnGroup-item" id="raw-url">Raw</a>
        <a href="/denizyuret/Knet.jl/blame/master/src/rnn.jl" class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b">Blame</a>
      <a href="/denizyuret/Knet.jl/commits/master/src/rnn.jl" class="btn btn-sm BtnGroup-item" rel="nofollow">History</a>
    </div>


        <button type="button" class="btn-octicon disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-pencil" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
        </button>
        <button type="button" class="btn-octicon btn-octicon-danger disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-trashcan" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
        </button>
  </div>

  <div class="file-info">
      747 lines (669 sloc)
      <span class="file-info-divider"></span>
    29.4 KB
  </div>
</div>

    

  <div itemprop="text" class="blob-wrapper data type-julia">
      <table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> TODO: </span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> finish cpu implementation.</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> make RNN objects callable?</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>## Size chart (Julia sizes for CUDNN calls)</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> Note: For Julia calls, x and y do not need the initial 1 dimension and B,T are optional.</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span></span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> x: (1,X,B,T) where X = inputSize, B = miniBatch, T = seqLength</span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> xDesc: Array of T (1,X,B) descriptors</span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> y: (1,Y,B,T) where Y = hiddenSize * (bidirectional ? 2 : 1)</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> yDesc: Array of T (1,Y,B) descriptors</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> w: (1,1,W) where W = cudnnGetRNNParamsSize()</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> hx,cx,hy,cy: (H,B,L) where H = hidden size, L = numLayers * (bidirectional ? 2 : 1)</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span></span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> Note: cudnn docs say min tensor dims 4 but RNN_example.cu uses 3D tensors</span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;</span>Dropout descriptor<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line"><span class="pl-k">type</span> DD; ptr<span class="pl-k">::</span><span class="pl-c1">Cptr</span>; states<span class="pl-k">::</span><span class="pl-c1">KnetArray{UInt8,1}</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">Base<span class="pl-k">.</span><span class="pl-en">unsafe_convert</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Cptr}</span>, dd<span class="pl-k">::</span><span class="pl-c1">DD</span>)<span class="pl-k">=</span>dd<span class="pl-k">.</span>ptr</td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">DD</span>(; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>(), dropout<span class="pl-k">=</span><span class="pl-c1">0.0</span>, seed<span class="pl-k">=</span><span class="pl-c1">0</span>, o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> seed<span class="pl-k">==</span><span class="pl-c1">0</span>; seed<span class="pl-k">=</span><span class="pl-c1">floor</span>(Culonglong,<span class="pl-c1">time</span>()); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">    d <span class="pl-k">=</span> Cptr[<span class="pl-c1">0</span>]; s <span class="pl-k">=</span> Csize_t[<span class="pl-c1">0</span>] <span class="pl-c"><span class="pl-c">#</span> TODO: Can multiple RNNs share dropout descriptors? Can dropout probability be changed?</span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn,cudnnCreateDropoutDescriptor,(Ptr{Cptr},),d)</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn,cudnnDropoutGetStatesSize,(Cptr,Ptr{Csize_t}),handle,s)</td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">    states <span class="pl-k">=</span> <span class="pl-c1">KnetArray</span><span class="pl-c1">{UInt8}</span>(s[<span class="pl-c1">1</span>]) <span class="pl-c"><span class="pl-c">#</span> TODO: Can this be shared? 638976 bytes.</span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn,cudnnSetDropoutDescriptor,(Cptr,Cptr,Cfloat,Cptr,Csize_t,Culonglong),</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">          d[<span class="pl-c1">1</span>],handle,dropout,states,<span class="pl-c1">bytes</span>(states),seed)</td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">    dd <span class="pl-k">=</span> <span class="pl-c1">DD</span>(d[<span class="pl-c1">1</span>],states)</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">finalizer</span>(dd, x<span class="pl-k">-&gt;</span><span class="pl-c1">@cuda</span>(cudnn,cudnnDestroyDropoutDescriptor,(Cptr,),x<span class="pl-k">.</span>ptr))</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> dd</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;</span>RNN descriptor<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line"><span class="pl-k">type</span> RD; ptr<span class="pl-k">::</span><span class="pl-c1">Cptr</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">Base<span class="pl-k">.</span><span class="pl-en">unsafe_convert</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Cptr}</span>, rd<span class="pl-k">::</span><span class="pl-c1">RD</span>)<span class="pl-k">=</span>rd<span class="pl-k">.</span>ptr</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">RD</span>()</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">    d <span class="pl-k">=</span> Cptr[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn,cudnnCreateRNNDescriptor,(Ptr{Cptr},),d)</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    rd <span class="pl-k">=</span> <span class="pl-c1">RD</span>(d[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">finalizer</span>(rd, x<span class="pl-k">-&gt;</span><span class="pl-c1">@cuda</span>(cudnn,cudnnDestroyRNNDescriptor,(Cptr,),x<span class="pl-k">.</span>ptr))</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> rd</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;</span>RNN config<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line"><span class="pl-k">type</span> RNN</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">    inputSize<span class="pl-k">::</span><span class="pl-c1">Cint</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">    hiddenSize<span class="pl-k">::</span><span class="pl-c1">Cint</span></td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">    numLayers<span class="pl-k">::</span><span class="pl-c1">Cint</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">    dropout<span class="pl-k">::</span><span class="pl-c1">Float64</span></td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">    inputMode<span class="pl-k">::</span><span class="pl-c1">Cint</span>    <span class="pl-c"><span class="pl-c">#</span> CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1    </span></td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">    direction<span class="pl-k">::</span><span class="pl-c1">Cint</span>    <span class="pl-c"><span class="pl-c">#</span> CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1</span></td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">    mode<span class="pl-k">::</span><span class="pl-c1">Cint</span>         <span class="pl-c"><span class="pl-c">#</span> CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3</span></td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">    algo<span class="pl-k">::</span><span class="pl-c1">Cint</span>         <span class="pl-c"><span class="pl-c">#</span> CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">    dataType<span class="pl-k">::</span><span class="pl-c1">DataType</span> <span class="pl-c"><span class="pl-c">#</span> CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2</span></td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">    rnnDesc<span class="pl-k">::</span><span class="pl-c1">Union{RD,Void}</span></td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    dropoutDesc<span class="pl-k">::</span><span class="pl-c1">Union{DD,Void}</span></td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">    dx</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">    dhx</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">    dcx</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line"><span class="pl-en">rnnids</span>(r) <span class="pl-k">=</span> (r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span> ? <span class="pl-c1">8</span> <span class="pl-k">:</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">3</span> ? <span class="pl-c1">6</span> <span class="pl-k">:</span> <span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">cudnnGetRNNParamsSize</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>())</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> r<span class="pl-k">.</span>rnnDesc <span class="pl-k">!=</span> <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">        res <span class="pl-k">=</span> Csize_t[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">        xDesc <span class="pl-k">=</span> <span class="pl-en">TD</span>(r<span class="pl-k">.</span>dataType, <span class="pl-c1">1</span>, r<span class="pl-k">.</span>inputSize, <span class="pl-c1">1</span>)    <span class="pl-c"><span class="pl-c">#</span> xDesc: (1,X,B) where X = inputSize, B is ignored, so assume 1</span></td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@cuda</span>(cudnn, cudnnGetRNNParamsSize,</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">              <span class="pl-c"><span class="pl-c">#</span> handle, rnndesc, xdesc, result, dataType</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">              (Cptr,  Cptr, Cptr, Ptr{Csize_t}, UInt32),</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">              handle, r<span class="pl-k">.</span>rnnDesc, xDesc, res, <span class="pl-c1">DT</span>(r<span class="pl-k">.</span>dataType))</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">div</span>(res[<span class="pl-c1">1</span>], <span class="pl-c1">sizeof</span>(r<span class="pl-k">.</span>dataType))</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span> <span class="pl-c"><span class="pl-c">#</span> on CPU, so we guess the size</span></td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">        X,H,L,I <span class="pl-k">=</span> r<span class="pl-k">.</span>inputSize, r<span class="pl-k">.</span>hiddenSize, r<span class="pl-k">.</span>numLayers, <span class="pl-c1">rnnids</span>(r)</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">        biases <span class="pl-k">=</span> L<span class="pl-k">*</span>I</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">        inputMatrices <span class="pl-k">=</span> (r<span class="pl-k">.</span>inputMode <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">0</span> <span class="pl-k">:</span> r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? I <span class="pl-k">:</span> <span class="pl-c1">div</span>(I,<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">        hiddenMatrices <span class="pl-k">=</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? (L<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">*</span>I <span class="pl-k">:</span> (L<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">*</span>I <span class="pl-k">+</span> <span class="pl-c1">div</span>(I,<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">        biases <span class="pl-k">*</span> H <span class="pl-k">+</span> inputMatrices <span class="pl-k">*</span> X <span class="pl-k">*</span> H <span class="pl-k">+</span> hiddenMatrices <span class="pl-k">*</span> H <span class="pl-k">*</span> H</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;</span>Keeps an array of 3D tensor descriptors<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line"><span class="pl-k">type</span> TDs; pvec<span class="pl-k">::</span><span class="pl-c1">Vector{Cptr}</span>; xDesc<span class="pl-k">::</span><span class="pl-c1">Vector{TD}</span>; <span class="pl-k">end</span>     <span class="pl-c"><span class="pl-c">#</span> Keep xDesc in TDs so it does not get gc&#39;ed</span></td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">Base<span class="pl-k">.</span><span class="pl-en">unsafe_convert</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Ptr{Cptr}}</span>, tds<span class="pl-k">::</span><span class="pl-c1">TDs</span>)<span class="pl-k">=</span><span class="pl-c1">pointer</span>(tds<span class="pl-k">.</span>pvec)</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">Base<span class="pl-k">.</span><span class="pl-en">length</span>(tds<span class="pl-k">::</span><span class="pl-c1">TDs</span>)<span class="pl-k">=</span><span class="pl-c1">length</span>(tds<span class="pl-k">.</span>pvec)</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">TDs</span><span class="pl-c1">{A}</span>(x<span class="pl-k">::</span><span class="pl-c1">KnetArray{A}</span>,<span class="pl-k">::</span><span class="pl-c1">Void</span>) <span class="pl-c"><span class="pl-c">#</span> Treat x: (X,B?,T?) as a 4D array: (1,X,B,T)</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">    xDesc <span class="pl-k">=</span> <span class="pl-c1">TD</span>(A,<span class="pl-c1">1</span>,<span class="pl-c1">size</span>(x,<span class="pl-c1">1</span>),<span class="pl-c1">size</span>(x,<span class="pl-c1">2</span>)) <span class="pl-c"><span class="pl-c">#</span> we can use a single xDesc</span></td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">    pvec <span class="pl-k">=</span> <span class="pl-c1">Vector</span><span class="pl-c1">{Cptr}</span>(<span class="pl-c1">size</span>(x,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">    pvec[<span class="pl-k">:</span>] <span class="pl-k">=</span> xDesc<span class="pl-k">.</span>ptr</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> <span class="pl-c1">TDs</span>(pvec, [xDesc])</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">TDs</span><span class="pl-c1">{A}</span>(x<span class="pl-k">::</span><span class="pl-c1">KnetArray{A}</span>,batchSizes) <span class="pl-c"><span class="pl-c">#</span> x: (X,B*), batchSizes gives us Bt sizes</span></td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@assert</span> <span class="pl-c1">sum</span>(batchSizes) <span class="pl-k">==</span> <span class="pl-c1">div</span>(<span class="pl-c1">length</span>(x),<span class="pl-c1">size</span>(x,<span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">    X <span class="pl-k">=</span> <span class="pl-c1">size</span>(x,<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">    xs <span class="pl-k">=</span> [ <span class="pl-c1">TD</span>(A,<span class="pl-c1">1</span>,X,B) <span class="pl-k">for</span> B <span class="pl-k">in</span> batchSizes ]</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">    ps <span class="pl-k">=</span> [ xd<span class="pl-k">.</span>ptr <span class="pl-k">for</span> xd <span class="pl-k">in</span> xs ]</td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> <span class="pl-c1">TDs</span>(ps,xs)</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">TD3</span>(a<span class="pl-k">::</span><span class="pl-c1">KnetArray</span>) <span class="pl-c"><span class="pl-c">#</span> Treat a as a 3D array, pad from right</span></td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">    n <span class="pl-k">=</span> <span class="pl-c1">ndims</span>(a)</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> n<span class="pl-k">==</span><span class="pl-c1">3</span>; <span class="pl-c1">TD</span>(a)</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> n<span class="pl-k">==</span><span class="pl-c1">2</span>; <span class="pl-c1">TD</span>(<span class="pl-c1">reshape</span>(a, <span class="pl-c1">size</span>(a,<span class="pl-c1">1</span>), <span class="pl-c1">size</span>(a,<span class="pl-c1">2</span>), <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> n<span class="pl-k">==</span><span class="pl-c1">1</span>; <span class="pl-c1">TD</span>(<span class="pl-c1">reshape</span>(a, <span class="pl-c1">size</span>(a,<span class="pl-c1">1</span>), <span class="pl-c1">1</span>, <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span>; <span class="pl-c1">throw</span>(<span class="pl-c1">DimensionMismatch</span>())</td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">FD3</span>(a<span class="pl-k">::</span><span class="pl-c1">KnetArray</span>) <span class="pl-c"><span class="pl-c">#</span> Treat a as a 3D array, pad from left</span></td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">    n <span class="pl-k">=</span> <span class="pl-c1">ndims</span>(a)</td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> n<span class="pl-k">==</span><span class="pl-c1">3</span>; <span class="pl-c1">FD</span>(a)</td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> n<span class="pl-k">==</span><span class="pl-c1">2</span>; <span class="pl-c1">FD</span>(<span class="pl-c1">reshape</span>(a, <span class="pl-c1">1</span>, <span class="pl-c1">size</span>(a,<span class="pl-c1">1</span>), <span class="pl-c1">size</span>(a,<span class="pl-c1">2</span>)))</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> n<span class="pl-k">==</span><span class="pl-c1">1</span>; <span class="pl-c1">FD</span>(<span class="pl-c1">reshape</span>(a, <span class="pl-c1">1</span>, <span class="pl-c1">1</span>, <span class="pl-c1">size</span>(a,<span class="pl-c1">1</span>)))</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span>; <span class="pl-c1">throw</span>(<span class="pl-c1">DimensionMismatch</span>())</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">cudnnGetRNNWorkspaceSize</span>(rd<span class="pl-k">::</span><span class="pl-c1">RD</span>, tds<span class="pl-k">::</span><span class="pl-c1">TDs</span>; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>())</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">    res <span class="pl-k">=</span> Csize_t[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn, cudnnGetRNNWorkspaceSize,</td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">          <span class="pl-c"><span class="pl-c">#</span> handle, rnndesc, seqLength, xdesc, res        ,</span></td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">          handle, rd, <span class="pl-c1">length</span>(tds), tds, res)</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> <span class="pl-c1">Int</span>(res[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">cudnnGetRNNTrainingReserveSize</span>(rd<span class="pl-k">::</span><span class="pl-c1">RD</span>, tds<span class="pl-k">::</span><span class="pl-c1">TDs</span>; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>())</td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">    res <span class="pl-k">=</span> Csize_t[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn, cudnnGetRNNTrainingReserveSize,</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">          <span class="pl-c"><span class="pl-c">#</span> handle, rnndesc, seqLength, xdesc, res        ,</span></td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">          (Cptr,  Cptr, Cint, Ptr{Cptr}, Ptr{Csize_t}),</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">          handle, rd, <span class="pl-c1">length</span>(tds), tds, res)</td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> <span class="pl-c1">Int</span>(res[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> Return eltype,size</span></td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">cudnnGetFilterNdDescriptor</span>(wDesc<span class="pl-k">::</span><span class="pl-c1">FD</span>; nbDimsRequested <span class="pl-k">=</span> <span class="pl-c1">8</span>)</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">    dataType <span class="pl-k">=</span> Cint[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">    format <span class="pl-k">=</span> Cint[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">    nbDims <span class="pl-k">=</span> Cint[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">    filterDimA <span class="pl-k">=</span> <span class="pl-c1">Vector</span><span class="pl-c1">{Cint}</span>(nbDimsRequested)</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn, cudnnGetFilterNdDescriptor,</td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">          (Cptr, Cint, Ptr{UInt32}, Ptr{UInt32}, Ptr{Cint}, Ptr{Cint}),</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">          wDesc, nbDimsRequested, dataType, format, nbDims, filterDimA)</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> nbDims[<span class="pl-c1">1</span>] <span class="pl-k">&gt;</span> nbDimsRequested</td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cudnnGetFilterNdDescriptor</span>(wDesc<span class="pl-k">::</span><span class="pl-c1">FD</span>; nbDimsRequested <span class="pl-k">=</span> nbDims[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span></td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">        (Float32,Float64,Float16)[<span class="pl-c1">1</span><span class="pl-k">+</span>dataType[<span class="pl-c1">1</span>]],</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">        (filterDimA[nbDims[<span class="pl-c1">1</span>]<span class="pl-k">:</span><span class="pl-k">-</span><span class="pl-c1">1</span><span class="pl-k">:</span><span class="pl-c1">1</span>]<span class="pl-k">.</span><span class="pl-k">..</span>)</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">    rnnparam{T}(r::RNN, w::KnetArray{T}, layer, id, param)</span></span></td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Return a single weight matrix or bias vector as a slice of w.</span></span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Valid <span class="pl-c1">`layer`</span> values:</span></span></td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For unidirectional RNNs 1:numLayers</span></span></td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For bidirectional RNNs 1:2*numLayers, forw and back layers alternate.</span></span></td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Valid <span class="pl-c1">`id`</span> values:</span></span></td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For RELU and TANH RNNs, input = 1, hidden = 2.</span></span></td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For GRU reset = 1,4; update = 2,5; newmem = 3,6; 1:3 for input, 4:6 for hidden</span></span></td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For LSTM inputgate = 1,5; forget = 2,6; newmem = 3,7; output = 4,8; 1:4 for input, 5:8 for hidden</span></span></td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Valid <span class="pl-c1">`param`</span> values:</span></span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> Return the weight matrix (transposed!) if <span class="pl-c1">`param==1`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> Return the bias vector if <span class="pl-c1">`param==2`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">The effect of skipInput: Let I=1 for RELU/TANH, 1:3 for GRU, 1:4 for LSTM</span></span></td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For skipInput=false (default), rnnparam(r,w,1,I,1) is a (inputSize,hiddenSize) matrix.</span></span></td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For skipInput=true, rnnparam(r,w,1,I,1) is <span class="pl-c1">`nothing`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> For bidirectional, the same applies to rnnparam(r,w,2,I,1): the first back layer.</span></span></td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnparam</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w, layer<span class="pl-k">::</span><span class="pl-c1">Integer</span>, id<span class="pl-k">::</span><span class="pl-c1">Integer</span>, par<span class="pl-k">::</span><span class="pl-c1">Integer</span>; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>())</td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> w could be a Rec, KnetArray, or Array so typing w::KnetArray{T} is not an option</span></td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">    ((<span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> par <span class="pl-k">&lt;=</span> <span class="pl-c1">2</span>) <span class="pl-k">&amp;&amp;</span></td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">     ((r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> layer <span class="pl-k">&lt;=</span> r<span class="pl-k">.</span>numLayers) <span class="pl-k">||</span></td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">      (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> layer <span class="pl-k">&lt;=</span> <span class="pl-c1">2</span><span class="pl-k">*</span>r<span class="pl-k">.</span>numLayers)) <span class="pl-k">&amp;&amp;</span></td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">     ((r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> id <span class="pl-k">&lt;=</span> <span class="pl-c1">2</span>) <span class="pl-k">||</span></td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">      (r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> id <span class="pl-k">&lt;=</span> <span class="pl-c1">2</span>) <span class="pl-k">||</span></td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">      (r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> id <span class="pl-k">&lt;=</span> <span class="pl-c1">8</span>) <span class="pl-k">||</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">      (r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">3</span> <span class="pl-k">&amp;&amp;</span> <span class="pl-c1">1</span> <span class="pl-k">&lt;=</span> id <span class="pl-k">&lt;=</span> <span class="pl-c1">6</span>))) <span class="pl-k">||</span> <span class="pl-c1">error</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>Bad parameter index<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> <span class="pl-c1">isa</span>(<span class="pl-c1">getval</span>(w), KnetArray)</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> <span class="pl-c1">eltype</span>(w)</td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">        xDesc <span class="pl-k">=</span> <span class="pl-c1">TD</span>(T,<span class="pl-c1">1</span>,r<span class="pl-k">.</span>inputSize,<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">        wDesc <span class="pl-k">=</span> <span class="pl-c1">FD</span>(T,<span class="pl-c1">1</span>,<span class="pl-c1">1</span>,<span class="pl-c1">length</span>(w))</td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">        paramDesc <span class="pl-k">=</span> <span class="pl-c1">FD</span>(T,<span class="pl-c1">1</span>,<span class="pl-c1">1</span>,<span class="pl-c1">1</span>,<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">        param <span class="pl-k">=</span> Cptr[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> par <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-c"><span class="pl-c">#</span> matrix</span></td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">@cuda</span>(cudnn, cudnnGetRNNLinLayerMatrixParams,</td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">                  (Cptr, Cptr, Cint, <span class="pl-c"><span class="pl-c">#</span>handle,rdesc, layer</span></td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">                   Cptr, Cptr, Cptr, <span class="pl-c"><span class="pl-c">#</span>xDesc, wDesc, w</span></td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">                   Cint, Cptr, Ptr{Cptr}), <span class="pl-c"><span class="pl-c">#</span>lid, lmatdesc, linlayermat</span></td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">                  handle, r<span class="pl-k">.</span>rnnDesc, layer<span class="pl-k">-</span><span class="pl-c1">1</span>,</td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">                  xDesc, wDesc, <span class="pl-c1">getval</span>(w),</td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">                  id<span class="pl-k">-</span><span class="pl-c1">1</span>, paramDesc, param)</td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span> <span class="pl-c"><span class="pl-c">#</span> bias</span></td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">@cuda</span>(cudnn, cudnnGetRNNLinLayerBiasParams,</td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">                  (Cptr, Cptr, Cint, <span class="pl-c"><span class="pl-c">#</span>handle,rdesc, layer</span></td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">                   Cptr, Cptr, Cptr, <span class="pl-c"><span class="pl-c">#</span>xDesc, wDesc, w</span></td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">                   Cint, Cptr, Ptr{Cptr}), <span class="pl-c"><span class="pl-c">#</span>lid, lmatdesc, linlayermat</span></td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">                  handle, r<span class="pl-k">.</span>rnnDesc, layer<span class="pl-k">-</span><span class="pl-c1">1</span>,</td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">                  xDesc, wDesc, <span class="pl-c1">getval</span>(w),</td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">                  id<span class="pl-k">-</span><span class="pl-c1">1</span>, paramDesc, param)</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">        dt,sz <span class="pl-k">=</span> <span class="pl-c1">cudnnGetFilterNdDescriptor</span>(paramDesc)</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">        len <span class="pl-k">=</span> <span class="pl-c1">prod</span>(sz)</td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">div</span>(<span class="pl-c1">Int</span>(param[<span class="pl-c1">1</span>] <span class="pl-k">-</span> <span class="pl-c1">pointer</span>(w)), <span class="pl-c1">sizeof</span>(T))</td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> i1 <span class="pl-k">+</span> len <span class="pl-k">-</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span></td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> guess i1,i2,len from layer,id,par and rnn specs</span></td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">        ids <span class="pl-k">=</span> <span class="pl-c1">rnnids</span>(r)</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> par <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-c"><span class="pl-c">#</span> matrix</span></td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> input matrices are (inputSize,hiddenSize) all others (hiddenSize,hiddenSize)</span></td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> if inputMode==1 then input matrices are empty</span></td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> I=1 for RELU/TANH, 1:3 for GRU, 1:4 for LSTM are input matrices with L=1 for uni and L=1,2 for bi</span></td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">            <span class="pl-en">inputLayer</span>(r,l)<span class="pl-k">=</span>(l<span class="pl-k">==</span><span class="pl-c1">1</span> <span class="pl-k">||</span> (l<span class="pl-k">==</span><span class="pl-c1">2</span> <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>direction<span class="pl-k">==</span><span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">            X, H <span class="pl-k">=</span> r<span class="pl-k">.</span>inputSize, r<span class="pl-k">.</span>hiddenSize</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">            XH, HH <span class="pl-k">=</span> X<span class="pl-k">*</span>H, H<span class="pl-k">*</span>H</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">            inputIds <span class="pl-k">=</span> <span class="pl-c1">div</span>(ids,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">            i1 <span class="pl-k">=</span> i2 <span class="pl-k">=</span> len <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> l <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>layer, i <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>ids</td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">inputLayer</span>(r,l) <span class="pl-k">&amp;&amp;</span> i <span class="pl-k">&lt;=</span> inputIds</td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">                    len <span class="pl-k">=</span> (r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? XH <span class="pl-k">:</span> <span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span></td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">                    len <span class="pl-k">=</span> HH</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">                i2 <span class="pl-k">+=</span> len</td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> l<span class="pl-k">==</span>layer <span class="pl-k">&amp;&amp;</span> i<span class="pl-k">==</span>id; <span class="pl-k">break</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> len<span class="pl-k">==</span><span class="pl-c1">0</span>; len<span class="pl-k">=</span><span class="pl-c1">1</span>; <span class="pl-k">end</span> <span class="pl-c"><span class="pl-c">#</span> cudnn uses size (1,1,1) for empty weights</span></td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">            i1 <span class="pl-k">=</span> i2 <span class="pl-k">-</span> len <span class="pl-k">+</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span> <span class="pl-c"><span class="pl-c">#</span> bias</span></td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> all biases are length=hidden and there are always numLayers * numIds of them</span></td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">            len <span class="pl-k">=</span> r<span class="pl-k">.</span>hiddenSize</td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">            layers <span class="pl-k">=</span> r<span class="pl-k">.</span>numLayers <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">            i1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">length</span>(w) <span class="pl-k">-</span> layers <span class="pl-k">*</span> ids <span class="pl-k">*</span> len <span class="pl-k">+</span> ((id<span class="pl-k">-</span><span class="pl-c1">1</span>) <span class="pl-k">+</span> ids <span class="pl-k">*</span> (layer<span class="pl-k">-</span><span class="pl-c1">1</span>)) <span class="pl-k">*</span> len</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">            i2 <span class="pl-k">=</span> i1 <span class="pl-k">+</span> len <span class="pl-k">-</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> len <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-c"><span class="pl-c">#</span> empty weights when inputMode=1 show up as size (1,1,1)</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> par <span class="pl-k">==</span> <span class="pl-c1">1</span> <span class="pl-c"><span class="pl-c">#</span> matrix</span></td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">        h <span class="pl-k">=</span> <span class="pl-c1">Int</span>(r<span class="pl-k">.</span>hiddenSize)</td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">reshape</span>(w[i1<span class="pl-k">:</span>i2], (<span class="pl-c1">div</span>(len,h),h)) <span class="pl-c"><span class="pl-c">#</span> weight matrices are transposed</span></td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span> <span class="pl-c"><span class="pl-c">#</span> bias</span></td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">        w[i1<span class="pl-k">:</span>i2]</td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">    rnnparams(r::RNN, w)</span></span></td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Split w into individual parameters and return them as an array.</span></span></td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">The order of params returned (subject to change):</span></span></td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> All weight matrices come before all bias vectors.</span></span></td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> Matrices and biases are sorted lexically based on (layer,id).</span></span></td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> See <span class="pl-v">@</span><span class="pl-s">doc</span> rnnparam for valid layer and id values.</span></span></td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">*</span> Input multiplying matrices are <span class="pl-c1">`nothing`</span> if r.inputMode = 1.</span></span></td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnparams</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>())</td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">    layers <span class="pl-k">=</span> r<span class="pl-k">.</span>numLayers <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">    ids <span class="pl-k">=</span> <span class="pl-c1">rnnids</span>(r)</td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">    ws <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> m <span class="pl-k">in</span> (<span class="pl-c1">1</span>,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> l <span class="pl-k">in</span> <span class="pl-c1">1</span><span class="pl-k">:</span>layers</td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">1</span><span class="pl-k">:</span>ids</td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">push!</span>(ws, <span class="pl-c1">rnnparam</span>(r, w, l, i, m; handle<span class="pl-k">=</span>handle))</td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> ws</td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-pds">&quot;&quot;&quot;</span><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">    rnninit(inputSize, hiddenSize; opts...)</span></span></td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">Return an <span class="pl-c1">`(r,w)`</span> pair where <span class="pl-c1">`r`</span> is a RNN struct and <span class="pl-c1">`w`</span> is a single weight</span></span></td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">array that includes all matrices and biases for the RNN. Keyword arguments:</span></span></td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`rnnType=:lstm`</span> Type of RNN: One of :relu, :tanh, :lstm, :gru.</span></span></td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`numLayers=1`</span>: Number of RNN layers.</span></span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`bidirectional=false`</span>: Create a bidirectional RNN if <span class="pl-c1">`true`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`dropout=0.0`</span>: Dropout probability. Ignored if <span class="pl-c1">`numLayers==1`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`skipInput=false`</span>: Do not multiply the input with a matrix if <span class="pl-c1">`true`</span>.</span></span></td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`dataType=Float32`</span>: Data type to use for weights.</span></span></td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`algo=0`</span>: Algorithm to use, see CUDNN docs for details.</span></span></td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`seed=0`</span>: Random number seed. Uses <span class="pl-c1">`time()`</span> if 0.</span></span></td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`winit=xavier`</span>: Weight initialization method for matrices.</span></span></td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`binit=zeros`</span>: Weight initialization method for bias vectors.</span></span></td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`usegpu=(gpu()&gt;=0): GPU used by default if one exists.</span></span></span></td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">RNNs compute the output h[t] for a given iteration from the recurrent</span></span></span></td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">input h[t-1] and the previous layer input x[t] given matrices W, R and</span></span></span></td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">biases bW, bR from the following equations:</span></span></span></td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`</span>:relu<span class="pl-c1">` and `</span>:tanh<span class="pl-c1">`: Single gate RNN with activation function f:</span></span></span></td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    h[t] = f(W * x[t] .+ R * h[t-1] .+ bW .+ bR)</span></span></span></td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`</span>:gru<span class="pl-c1">`: Gated recurrent unit:</span></span></span></td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    i[t] = sigm(Wi * x[t] .+ Ri * h[t-1] .+ bWi .+ bRi) # input gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    r[t] = sigm(Wr * x[t] .+ Rr * h[t-1] .+ bWr .+ bRr) # reset gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    n[t] = tanh(Wn * x[t] .+ r[t] .* (Rn * h[t-1] .+ bRn) .+ bWn) # new gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    h[t] = (1 - i[t]) .* n[t] .+ i[t] .* h[t-1]</span></span></span></td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`</span>:lstm<span class="pl-c1">`: Long short term memory unit with no peephole connections:</span></span></span></td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    i[t] = sigm(Wi * x[t] .+ Ri * h[t-1] .+ bWi .+ bRi) # input gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    f[t] = sigm(Wf * x[t] .+ Rf * h[t-1] .+ bWf .+ bRf) # forget gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    o[t] = sigm(Wo * x[t] .+ Ro * h[t-1] .+ bWo .+ bRo) # output gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    n[t] = tanh(Wn * x[t] .+ Rn * h[t-1] .+ bWn .+ bRn) # new gate</span></span></span></td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    c[t] = f[t] .* c[t-1] .+ i[t] .* n[t]               # cell output</span></span></span></td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    h[t] = o[t] .* tanh(c[t])</span></span></span></td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">&quot;&quot;&quot;</span></span></span></td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">function rnninit(inputSize, hiddenSize;</span></span></span></td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 handle=cudnnhandle(),</span></span></span></td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 numLayers=1,</span></span></span></td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 dropout=0.0,</span></span></span></td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1</span></span></span></td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1</span></span></span></td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3</span></span></span></td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 dataType=Float32,    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2</span></span></span></td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2</span></span></span></td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 seed=0,              # seed=0 for random init, positive integer for replicability</span></span></span></td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 winit=xavier,</span></span></span></td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 binit=zeros,</span></span></span></td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 usegpu=(gpu()&gt;=0),</span></span></span></td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                 )</span></span></span></td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    inputMode = skipInput ? 1 : 0</span></span></span></td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    direction = bidirectional ? 1 : 0</span></span></span></td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    mode = findfirst((:relu,:tanh,:lstm,:gru), rnnType) - 1</span></span></span></td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    if mode &lt; 0; error(&quot;rnninit: Valid modes are :relu,:tanh,:lstm,:gru&quot;); end</span></span></span></td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    if usegpu</span></span></span></td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        rnnDesc = RD()</span></span></span></td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        dropoutDesc = DD(handle=handle,dropout=dropout,seed=seed) # Need to keep dropoutDesc in RNN so it does not get gc&#39;ed.</span></span></span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        if cudnnVersion &gt;= 7000</span></span></span></td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),</span></span></span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                  handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))</span></span></span></td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        elseif cudnnVersion &gt;= 6000</span></span></span></td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            @cuda(cudnn,cudnnSetRNNDescriptor_v6,(Cptr,Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint,Cint),</span></span></span></td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                  handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,DT(dataType))</span></span></span></td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        elseif cudnnVersion &gt;= 5000</span></span></span></td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            @cuda(cudnn,cudnnSetRNNDescriptor,(Cptr,Cint,Cint,Cptr,Cint,Cint,Cint,Cint),</span></span></span></td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">                  rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,DT(dataType))</span></span></span></td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        else</span></span></span></td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            error(&quot;CUDNN $cudnnVersion does not support RNNs&quot;)</span></span></span></td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        end</span></span></span></td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        r = RNN(inputSize,hiddenSize,numLayers,dropout,inputMode,direction,mode,algo,dataType,rnnDesc,dropoutDesc,nothing,nothing,nothing)</span></span></span></td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        w = KnetArray{dataType}(1,1,cudnnGetRNNParamsSize(r))</span></span></span></td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    else</span></span></span></td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        r = RNN(inputSize,hiddenSize,numLayers,dropout,inputMode,direction,mode,algo,dataType,nothing,nothing,nothing,nothing,nothing)</span></span></span></td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        w = Array{dataType}(1,1,cudnnGetRNNParamsSize(r))</span></span></span></td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    end</span></span></span></td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    for a in rnnparams(r,w; handle=handle)</span></span></span></td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        if a == nothing</span></span></span></td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            continue</span></span></span></td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        elseif ndims(a) == 2</span></span></span></td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            copy!(a, winit(dataType, size(a)))</span></span></span></td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        elseif ndims(a) == 1</span></span></span></td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            copy!(a, binit(dataType, size(a)))</span></span></span></td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        else</span></span></span></td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">            error()</span></span></span></td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">        end</span></span></span></td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    end</span></span></span></td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    return (r,w)</span></span></span></td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">end</span></span></span></td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">&quot;&quot;&quot;</span></span></span></td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">    rnnforw(r, w, x[, hx, cx]; batchSizes, hy, cy)</span></span></span></td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1"></span></span></span></td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">Returns a tuple (y,hyout,cyout,rs) given rnn `</span>r<span class="pl-c1">`, weights `</span>w<span class="pl-c1">`, input</span></span></span></td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`</span>x<span class="pl-c1">` and optionally the initial hidden and cell states `</span>hx<span class="pl-c1">` and `</span>cx`</span></span></td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">(<span class="pl-c1">`cx`</span> is only used in LSTMs).  <span class="pl-c1">`r`</span> and <span class="pl-c1">`w`</span> should come from a previous</span></span></td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">call to <span class="pl-c1">`rnninit`</span>.  Both <span class="pl-c1">`hx`</span> and <span class="pl-c1">`cx`</span> are optional, they are treated</span></span></td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">as zero arrays if not provided.  The output <span class="pl-c1">`y`</span> contains the hidden</span></span></td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">states of the final layer for each time step, <span class="pl-c1">`hyout`</span> and <span class="pl-c1">`cyout`</span> give</span></span></td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">the final hidden and cell states for all layers, <span class="pl-c1">`rs`</span> is a buffer the</span></span></td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">RNN needs for its gradient calculation.</span></span></td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">The boolean keyword arguments <span class="pl-c1">`hy`</span> and <span class="pl-c1">`cy`</span> control whether <span class="pl-c1">`hyout`</span></span></span></td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">and <span class="pl-c1">`cyout`</span> will be output.  By default <span class="pl-c1">`hy = (hx!=nothing)`</span> and <span class="pl-c1">`cy =</span></span></span></td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">(cx!=nothing &amp;&amp; r.mode==2)`</span>, i.e. a hidden state will be output if one</span></span></td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">is provided as input and for cell state we also require an LSTM.  If</span></span></td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`hy`</span>/<span class="pl-c1">`cy`</span> is <span class="pl-c1">`false`</span>, <span class="pl-c1">`hyout`</span>/<span class="pl-c1">`cyout`</span> will be <span class="pl-c1">`nothing`</span>. <span class="pl-c1">`batchSizes`</span></span></span></td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">can be an integer array that specifies non-uniform batch sizes as</span></span></td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">explained below. By default <span class="pl-c1">`batchSizes=nothing`</span> and the same batch</span></span></td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">size, <span class="pl-c1">`size(x,2)`</span>, is used for all time steps.</span></span></td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">The input and output dimensions are:</span></span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`x`</span>: (X,[B,T])</span></span></td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`y`</span>: (H/2H,[B,T])</span></span></td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`hx`</span>,<span class="pl-c1">`cx`</span>,<span class="pl-c1">`hyout`</span>,<span class="pl-c1">`cyout`</span>: (H,B,L/2L)</span></span></td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-v">-</span> <span class="pl-c1">`batchSizes`</span>: <span class="pl-c1">`nothing`</span> or <span class="pl-c1">`Vector{Int}(T)`</span></span></span></td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">where X is inputSize, H is hiddenSize, B is batchSize, T is seqLength,</span></span></td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">L is numLayers.  <span class="pl-c1">`x`</span> can be 1, 2, or 3 dimensional.  If</span></span></td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`batchSizes==nothing`</span>, a 1-D <span class="pl-c1">`x`</span> represents a single instance, a 2-D</span></span></td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`x`</span> represents a single minibatch, and a 3-D <span class="pl-c1">`x`</span> represents a sequence</span></span></td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">of identically sized minibatches.  If <span class="pl-c1">`batchSizes`</span> is an array of</span></span></td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">(non-increasing) integers, it gives us the batch size for each time</span></span></td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">step in the sequence, in which case <span class="pl-c1">`sum(batchSizes)`</span> should equal</span></span></td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`div(length(x),size(x,1))`</span>. <span class="pl-c1">`y`</span> has the same dimensionality as <span class="pl-c1">`x`</span>,</span></span></td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">differing only in its first dimension, which is H if the RNN is</span></span></td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">unidirectional, 2H if bidirectional.  Hidden vectors <span class="pl-c1">`hx`</span>, <span class="pl-c1">`cx`</span>,</span></span></td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"><span class="pl-c1">`hyout`</span>, <span class="pl-c1">`cyout`</span> all have size (H,B1,L) for unidirectional RNNs, and</span></span></td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">(H,B1,2L) for bidirectional RNNs where B1 is the size of the first</span></span></td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1">minibatch.</span></span></td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span></span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line"><span class="pl-s"><span class="pl-s1"></span><span class="pl-pds">&quot;&quot;&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnforw</span><span class="pl-c1">{T}</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w<span class="pl-k">::</span><span class="pl-c1">KnetArray{T}</span>, x<span class="pl-k">::</span><span class="pl-c1">KnetArray{T}</span>,</td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">                    hx<span class="pl-k">::</span><span class="pl-c1">Union{KnetArray{T},Void}</span><span class="pl-k">=</span><span class="pl-c1">nothing</span>,</td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">                    cx<span class="pl-k">::</span><span class="pl-c1">Union{KnetArray{T},Void}</span><span class="pl-k">=</span><span class="pl-c1">nothing</span>;</td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">                    handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>(), training<span class="pl-k">=</span><span class="pl-c1">false</span>,</td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">                    batchSizes<span class="pl-k">=</span><span class="pl-c1">nothing</span>,</td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">                    hy <span class="pl-k">=</span> (hx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span>),</td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line">                    cy <span class="pl-k">=</span> (cx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span> <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span>),</td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line">                    )</td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Input descriptors</span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line">    seqLength <span class="pl-k">=</span> batchSizes<span class="pl-k">==</span><span class="pl-c1">nothing</span> ? <span class="pl-c1">size</span>(x,<span class="pl-c1">3</span>) <span class="pl-k">:</span> <span class="pl-c1">length</span>(batchSizes) <span class="pl-c"><span class="pl-c">#</span> (X,B,T) or (X,B+) with batchSizes</span></td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line">    wDesc <span class="pl-k">=</span> <span class="pl-c1">FD3</span>(w)              <span class="pl-c"><span class="pl-c">#</span> (1,1,W)</span></td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line">    xtds <span class="pl-k">=</span> <span class="pl-c1">TDs</span>(x,batchSizes)    <span class="pl-c"><span class="pl-c">#</span> (1,X,Bt) x T</span></td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> hx<span class="pl-k">==</span><span class="pl-c1">nothing</span>; hx<span class="pl-k">=</span>hxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; hxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(hx); <span class="pl-k">end</span> <span class="pl-c"><span class="pl-c">#</span> (H,B,L/2L)</span></td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> cx<span class="pl-k">==</span><span class="pl-c1">nothing</span> <span class="pl-k">||</span> r<span class="pl-k">.</span>mode <span class="pl-k">!=</span> <span class="pl-c1">2</span>; cx<span class="pl-k">=</span>cxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; cxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(cx); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Output arrays and descriptors</span></td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line">    ysize <span class="pl-k">=</span> <span class="pl-c1">collect</span>(<span class="pl-c1">size</span>(x))</td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line">    ysize[<span class="pl-c1">1</span>] <span class="pl-k">=</span> r<span class="pl-k">.</span>hiddenSize <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line">    y <span class="pl-k">=</span> <span class="pl-c1">similar</span>(x, ysize<span class="pl-k">...</span>)    <span class="pl-c"><span class="pl-c">#</span> (H/2H,B,T) or (H/2H,B+) -- y mirrors x except for the first dimension</span></td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line">    ytds <span class="pl-k">=</span> <span class="pl-c1">TDs</span>(y,batchSizes)    <span class="pl-c"><span class="pl-c">#</span> (1,H/2H,Bt) x T</span></td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Optionally output hidden and cell of last step</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line">    hyout <span class="pl-k">=</span> hyDesc <span class="pl-k">=</span> cyout <span class="pl-k">=</span> cyDesc <span class="pl-k">=</span> C_NULL</td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> hy <span class="pl-k">||</span> cy</td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">        firstBatchSize <span class="pl-k">=</span> batchSizes<span class="pl-k">==</span><span class="pl-c1">nothing</span> ? <span class="pl-c1">size</span>(x,<span class="pl-c1">2</span>) <span class="pl-k">:</span> batchSizes[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">        hsize <span class="pl-k">=</span> Int[r<span class="pl-k">.</span>hiddenSize, firstBatchSize, r<span class="pl-k">.</span>numLayers <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>)] <span class="pl-c"><span class="pl-c">#</span> (H,B,L/2L)</span></td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> hy; hyout<span class="pl-k">=</span><span class="pl-c1">similar</span>(y,hsize<span class="pl-k">...</span>); hyDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(hyout); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> cy <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>mode<span class="pl-k">==</span><span class="pl-c1">2</span>; cyout<span class="pl-k">=</span><span class="pl-c1">similar</span>(y,hsize<span class="pl-k">...</span>); cyDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(cyout); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> workSpace and reserveSpace</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line">    wss <span class="pl-k">=</span> <span class="pl-c1">cudnnGetRNNWorkspaceSize</span>(r<span class="pl-k">.</span>rnnDesc, xtds; handle<span class="pl-k">=</span>handle)</td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">    ws <span class="pl-k">=</span> <span class="pl-c1">cudnnWorkSpace</span>(wss)</td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> training</td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">        rss <span class="pl-k">=</span> <span class="pl-c1">cudnnGetRNNTrainingReserveSize</span>(r<span class="pl-k">.</span>rnnDesc, xtds; handle<span class="pl-k">=</span>handle)</td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">        rs <span class="pl-k">=</span> <span class="pl-c1">KnetArray</span><span class="pl-c1">{UInt8}</span>(rss)</td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@cuda</span>(cudnn, cudnnRNNForwardTraining,</td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">              (Cptr, Cptr, Cint,  <span class="pl-c"><span class="pl-c">#</span> handle,rnnDesc,seqLength</span></td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">               Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>x</span></td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>hx</span></td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>cx</span></td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>w</span></td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">               Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>y</span></td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>hy</span></td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>cy</span></td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">               Cptr, Csize_t, <span class="pl-c"><span class="pl-c">#</span>ws</span></td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">               Cptr ,Csize_t<span class="pl-c"><span class="pl-c">#</span>rs</span></td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">               ),</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">              handle, r<span class="pl-k">.</span>rnnDesc, seqLength,</td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">              xtds, x,</td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">              hxDesc, hx,</td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">              cxDesc, cx,</td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">              wDesc, w,</td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">              ytds, y,</td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">              hyDesc, hyout,</td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">              cyDesc, cyout,</td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">              ws, wss,</td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">              rs, rss)</td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span></td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">        rs <span class="pl-k">=</span> <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@cuda</span>(cudnn, cudnnRNNForwardInference,</td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">              (Cptr, Cptr, Cint,  <span class="pl-c"><span class="pl-c">#</span> handle,rnnDesc,seqLength</span></td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">               Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>x</span></td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>h</span></td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>c</span></td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>w</span></td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">               Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>y</span></td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>hy</span></td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">               Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>cy</span></td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">               Cptr, Csize_t, <span class="pl-c"><span class="pl-c">#</span>ws</span></td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">               ),</td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">              handle, r<span class="pl-k">.</span>rnnDesc, seqLength,</td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line">              xtds, x,</td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line">              hxDesc, hx,</td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">              cxDesc, cx,</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">              wDesc, w,</td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">              ytds, y,</td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">              hyDesc, hyout,</td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">              cyDesc, cyout,</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">              ws, wss)</td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> hyout <span class="pl-k">==</span> C_NULL; hyout <span class="pl-k">=</span> <span class="pl-c1">nothing</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> cyout <span class="pl-k">==</span> C_NULL; cyout <span class="pl-k">=</span> <span class="pl-c1">nothing</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> y, hyout, cyout, rs</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line"><span class="pl-en">rnnforw</span>(r<span class="pl-k">::</span><span class="pl-c1">Rec{RNN}</span>, w<span class="pl-k">...</span>; o<span class="pl-k">...</span>)<span class="pl-k">=</span><span class="pl-c1">rnnforw</span>(<span class="pl-c1">getval</span>(r), w<span class="pl-k">...</span>; o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line"><span class="pl-k">let</span> rnnforw_r <span class="pl-k">=</span> <span class="pl-c1">recorder</span>(rnnforw); <span class="pl-k">global</span> rnnforw</td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">rnnforw</span><span class="pl-c1">{K&lt;:KnetArray}</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w<span class="pl-k">::</span><span class="pl-c1">Rec{K}</span>, x<span class="pl-k">...</span>; o<span class="pl-k">...</span>)<span class="pl-k">=</span><span class="pl-c1">rnnforw_r</span>(r, w, x<span class="pl-k">...</span>; o<span class="pl-k">...</span>, training<span class="pl-k">=</span><span class="pl-c1">true</span>)</td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnforw</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Grad{2}}</span>, dt, t, r, w, x, hx<span class="pl-k">=</span><span class="pl-c1">nothing</span>, cx<span class="pl-k">=</span><span class="pl-c1">nothing</span>; o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">    y,hy,cy,rs <span class="pl-k">=</span> <span class="pl-c1">getval</span>(t)</td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">    dy,dhy,dcy,drs <span class="pl-k">=</span> <span class="pl-c1">getval</span>(dt)</td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">    w<span class="pl-k">=</span><span class="pl-c1">getval</span>(w); x<span class="pl-k">=</span><span class="pl-c1">getval</span>(x); hx<span class="pl-k">=</span><span class="pl-c1">getval</span>(hx); cx<span class="pl-k">=</span><span class="pl-c1">getval</span>(cx)</td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">rnnback</span>(r, w, x, y, dy, hx, cx, dhy, dcy, rs; o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line"><span class="pl-en">rnnforw</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Grad{3}}</span>, dt, t, r, w<span class="pl-k">...</span>; o<span class="pl-k">...</span>)<span class="pl-k">=</span>r<span class="pl-k">.</span>dx</td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line"><span class="pl-en">rnnforw</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Grad{4}}</span>, dt, t, r, w<span class="pl-k">...</span>; o<span class="pl-k">...</span>)<span class="pl-k">=</span>r<span class="pl-k">.</span>dhx</td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line"><span class="pl-en">rnnforw</span>(<span class="pl-k">::</span><span class="pl-c1">Type{Grad{5}}</span>, dt, t, r, w<span class="pl-k">...</span>; o<span class="pl-k">...</span>)<span class="pl-k">=</span>r<span class="pl-k">.</span>dcx</td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnback</span><span class="pl-c1">{T}</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w<span class="pl-k">::</span><span class="pl-c1">KnetArray{T}</span>, x<span class="pl-k">::</span><span class="pl-c1">KnetArray{T}</span>, y<span class="pl-k">::</span><span class="pl-c1">KnetArray{T}</span>,</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line">                    dy, hx, cx, dhy, dcy, rs; handle<span class="pl-k">=</span><span class="pl-c1">cudnnhandle</span>(), batchSizes<span class="pl-k">=</span><span class="pl-c1">nothing</span>, o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Input descriptors:</span></td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">    seqLength <span class="pl-k">=</span> batchSizes<span class="pl-k">==</span><span class="pl-c1">nothing</span> ? <span class="pl-c1">size</span>(x,<span class="pl-c1">3</span>) <span class="pl-k">:</span> <span class="pl-c1">length</span>(batchSizes) <span class="pl-c"><span class="pl-c">#</span> (X,B,T) or (X,B+) with batchSizes</span></td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">    wDesc <span class="pl-k">=</span> <span class="pl-c1">FD3</span>(w)              <span class="pl-c"><span class="pl-c">#</span> (1,1,W)</span></td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">    xtds <span class="pl-k">=</span> <span class="pl-c1">TDs</span>(x,batchSizes)    <span class="pl-c"><span class="pl-c">#</span> (X,B,T) -&gt; (1,X,B) x T</span></td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">    ytds <span class="pl-k">=</span> <span class="pl-c1">TDs</span>(y,batchSizes)    <span class="pl-c"><span class="pl-c">#</span> (H/2H,B,T) -&gt; (1,H/2H,B) x T</span></td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> dytds = TDs(dy,batchSizes)  # we use ytds for dytds</span></td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> dy <span class="pl-k">==</span> <span class="pl-c1">nothing</span>; dy<span class="pl-k">=</span><span class="pl-c1">zeros</span>(y); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> hx <span class="pl-k">==</span> <span class="pl-c1">nothing</span>; hx<span class="pl-k">=</span>hxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; hxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(hx); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> cx <span class="pl-k">==</span> <span class="pl-c1">nothing</span> <span class="pl-k">||</span> r<span class="pl-k">.</span>mode <span class="pl-k">!=</span> <span class="pl-c1">2</span>; cx<span class="pl-k">=</span>cxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; cxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(cx); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> dhy <span class="pl-k">==</span> <span class="pl-c1">nothing</span>; dhy<span class="pl-k">=</span>dhyDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; dhyDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(dhy); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> dcy <span class="pl-k">==</span> <span class="pl-c1">nothing</span> <span class="pl-k">||</span> r<span class="pl-k">.</span>mode <span class="pl-k">!=</span> <span class="pl-c1">2</span>; dcy<span class="pl-k">=</span>dcyDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; dcyDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(dcy); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Output arrays and descriptors:</span></td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">    dx <span class="pl-k">=</span> <span class="pl-c1">similar</span>(x)             <span class="pl-c"><span class="pl-c">#</span> (X,B,T) or (X,B+) with batchSizes</span></td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> dxtds = TDs(dx,batchSizes)  # we use xtds here</span></td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">    dw <span class="pl-k">=</span> <span class="pl-c1">zeros</span>(w)               <span class="pl-c"><span class="pl-c">#</span> dw is used additively, so we need zeros</span></td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">    dwDesc <span class="pl-k">=</span> <span class="pl-c1">FD3</span>(dw)</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> hx <span class="pl-k">==</span> C_NULL; dhx<span class="pl-k">=</span>dhxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; dhx<span class="pl-k">=</span><span class="pl-c1">similar</span>(hx); dhxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(dhx); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> cx <span class="pl-k">==</span> C_NULL; dcx<span class="pl-k">=</span>dcxDesc<span class="pl-k">=</span>C_NULL; <span class="pl-k">else</span>; dcx<span class="pl-k">=</span><span class="pl-c1">similar</span>(cx); dcxDesc<span class="pl-k">=</span><span class="pl-c1">TD3</span>(dcx); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> workSpace and reserveSpace</span></td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">    ws <span class="pl-k">=</span> <span class="pl-c1">cudnnWorkSpace</span>()</td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">    wss <span class="pl-k">=</span> <span class="pl-c1">bytes</span>(ws)</td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">    rss <span class="pl-k">=</span> <span class="pl-c1">bytes</span>(rs)</td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> data backward</span></td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn, cudnnRNNBackwardData,</td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">          (Cptr, Cptr, Cint,  <span class="pl-c"><span class="pl-c">#</span> handle, rnnDesc, seqLength</span></td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>y</span></td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dy</span></td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dhy</span></td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dcy</span></td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>w</span></td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>hx</span></td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>cx</span></td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dx</span></td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dhx</span></td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dcx</span></td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line">           Cptr, Csize_t, <span class="pl-c"><span class="pl-c">#</span>ws</span></td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line">           Cptr, Csize_t), <span class="pl-c"><span class="pl-c">#</span>rs</span></td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">          <span class="pl-c"><span class="pl-c">#</span> Use rtd with nullables</span></td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">          handle, r<span class="pl-k">.</span>rnnDesc, seqLength,</td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">          ytds, y,</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">          ytds, dy,</td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">          dhyDesc, dhy,</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">          dcyDesc, dcy,</td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">          wDesc, w,</td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">          hxDesc, hx,</td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">          cxDesc, cx,</td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line">          xtds, dx,</td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line">          dhxDesc, dhx,</td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">          dcxDesc, dcx,</td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">          ws, wss,</td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">          rs, rss)</td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> weights backward</span></td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@cuda</span>(cudnn, cudnnRNNBackwardWeights,</td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line">          (Cptr, Cptr, Cint,  <span class="pl-c"><span class="pl-c">#</span> handle, rnnDesc, seqLength</span></td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>x</span></td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>hx</span></td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>y</span></td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">           Cptr, Csize_t, <span class="pl-c"><span class="pl-c">#</span>ws</span></td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">           Cptr, Ptr{T}, <span class="pl-c"><span class="pl-c">#</span>dw</span></td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">           Ptr{Cptr}, Csize_t), <span class="pl-c"><span class="pl-c">#</span>rs</span></td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">          handle, r<span class="pl-k">.</span>rnnDesc, seqLength,</td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">          xtds, x,</td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">          hxDesc, hx,</td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">          ytds, y,</td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line">          ws, wss,</td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line">          dwDesc, dw,</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">          rs, rss)</td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Update the cache</span></td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> dhx<span class="pl-k">==</span>C_NULL; dhx<span class="pl-k">=</span><span class="pl-c1">nothing</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> dcx<span class="pl-k">==</span>C_NULL; dcx<span class="pl-k">=</span><span class="pl-c1">nothing</span>; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">    r<span class="pl-k">.</span>dx, r<span class="pl-k">.</span>dhx, r<span class="pl-k">.</span>dcx <span class="pl-k">=</span> dx, dhx, dcx</td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> dw</td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> CPU version</span></td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnnforw</span><span class="pl-c1">{T}</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, w<span class="pl-k">::</span><span class="pl-c1">Array{T}</span>, x<span class="pl-k">::</span><span class="pl-c1">Array{T}</span>,</td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">                    hx<span class="pl-k">::</span><span class="pl-c1">Union{Array{T},Void}</span><span class="pl-k">=</span><span class="pl-c1">nothing</span>,</td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line">                    cx<span class="pl-k">::</span><span class="pl-c1">Union{Array{T},Void}</span><span class="pl-k">=</span><span class="pl-c1">nothing</span>;</td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c"><span class="pl-c">#</span> handle=cudnnhandle(), training=false,</span></td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line">                    batchSizes<span class="pl-k">=</span><span class="pl-c1">nothing</span>,</td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line">                    hy <span class="pl-k">=</span> (hx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span>),</td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line">                    cy <span class="pl-k">=</span> (cx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span> <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span>),</td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">                    o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">rnntest</span>(r,w,x,hx,cx;batchSizes<span class="pl-k">=</span>batchSizes,hy<span class="pl-k">=</span>hy,cy<span class="pl-k">=</span>cy)</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> non-CUDNN cpu/gpu version</span></td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">rnntest</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>, ws, x, hx<span class="pl-k">=</span><span class="pl-c1">nothing</span>, cx<span class="pl-k">=</span><span class="pl-c1">nothing</span>;</td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line">                 batchSizes<span class="pl-k">=</span><span class="pl-c1">nothing</span>,</td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">                 hy <span class="pl-k">=</span> (hx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span>),</td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">                 cy <span class="pl-k">=</span> (cx <span class="pl-k">!=</span> <span class="pl-c1">nothing</span> <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span>),</td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">                 o<span class="pl-k">...</span>)</td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span>; <span class="pl-c1">error</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>rnntest bidirectional not implemented yet<span class="pl-pds">&quot;</span></span>); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> r<span class="pl-k">.</span>dropout <span class="pl-k">!=</span> <span class="pl-c1">0</span>; <span class="pl-c1">error</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>rnntest dropout not implemented yet<span class="pl-pds">&quot;</span></span>); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> batchSizes <span class="pl-k">!=</span> <span class="pl-c1">nothing</span>; <span class="pl-c1">error</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>rnntest batchSizes not implemented yet<span class="pl-pds">&quot;</span></span>); <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line">    w <span class="pl-k">=</span> <span class="pl-c1">rnnparams</span>(r,ws)</td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">    X,B,T <span class="pl-k">=</span> (<span class="pl-c1">size</span>(x,i) <span class="pl-k">for</span> i<span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">:</span><span class="pl-c1">3</span>) <span class="pl-c"><span class="pl-c">#</span> ndims(x) may be 1,2 or 3</span></td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@assert</span> X <span class="pl-k">==</span> r<span class="pl-k">.</span>inputSize</td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line">    Y <span class="pl-k">=</span> <span class="pl-c1">Int</span>(r<span class="pl-k">.</span>hiddenSize <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">    ysize <span class="pl-k">=</span> <span class="pl-c1">ntuple</span>(i<span class="pl-k">-&gt;</span>(i<span class="pl-k">==</span><span class="pl-c1">1</span> ? Y <span class="pl-k">:</span> <span class="pl-c1">size</span>(x,i)), <span class="pl-c1">ndims</span>(x)) <span class="pl-c"><span class="pl-c">#</span> to match ndims(y) to ndims(x)</span></td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line">    H <span class="pl-k">=</span> <span class="pl-c1">Int</span>(r<span class="pl-k">.</span>hiddenSize)</td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@assert</span> (r<span class="pl-k">.</span>inputMode <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">||</span> H <span class="pl-k">==</span> X)</td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line">    L <span class="pl-k">=</span> <span class="pl-c1">Int</span>(r<span class="pl-k">.</span>numLayers <span class="pl-k">*</span> (r<span class="pl-k">.</span>direction <span class="pl-k">==</span> <span class="pl-c1">1</span> ? <span class="pl-c1">2</span> <span class="pl-k">:</span> <span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line">    hsize <span class="pl-k">=</span> (H,B,L)</td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@assert</span> hx <span class="pl-k">==</span> <span class="pl-c1">nothing</span> <span class="pl-k">||</span> <span class="pl-c1">size</span>(hx) <span class="pl-k">==</span> hsize</td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">@assert</span> cx <span class="pl-k">==</span> <span class="pl-c1">nothing</span> <span class="pl-k">||</span> <span class="pl-c1">size</span>(cx) <span class="pl-k">==</span> hsize</td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">    h <span class="pl-k">=</span> hx<span class="pl-k">==</span><span class="pl-c1">nothing</span> ? <span class="pl-c1">fill!</span>(<span class="pl-c1">similar</span>(x,hsize),<span class="pl-c1">0</span>) <span class="pl-k">:</span> hx</td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">    ys <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">    hs <span class="pl-k">=</span> [ h[<span class="pl-k">:</span>,<span class="pl-k">:</span>,l] <span class="pl-k">for</span> l<span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">:</span>L ]</td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> r<span class="pl-k">.</span>mode <span class="pl-k">&lt;=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@assert</span> r<span class="pl-k">.</span>inputMode <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">||</span> <span class="pl-c1">all</span>(w[<span class="pl-c1">1</span><span class="pl-k">:</span><span class="pl-c1">1</span><span class="pl-k">+</span>r<span class="pl-k">.</span>direction] <span class="pl-k">.==</span> <span class="pl-c1">nothing</span>)</td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ht = f(W_i * x_t + R_i h_t-1 + b_Wi + b_Ri)</span></td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">        f <span class="pl-k">=</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">0</span> ? relu <span class="pl-k">:</span> tanh</td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> t <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>T</td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> l <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>L</td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">                wx,wh,bx,bh <span class="pl-k">=</span> w[<span class="pl-c1">2</span>l<span class="pl-k">-</span><span class="pl-c1">1</span>],w[<span class="pl-c1">2</span>l],w[<span class="pl-c1">2</span>L<span class="pl-k">+</span><span class="pl-c1">2</span>l<span class="pl-k">-</span><span class="pl-c1">1</span>],w[<span class="pl-c1">2</span>L<span class="pl-k">+</span><span class="pl-c1">2</span>l]</td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">                wxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? wx<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? wx<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line">                hs[l] <span class="pl-k">=</span> <span class="pl-c1">f</span>.(wxt <span class="pl-k">.+</span> wh<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bx <span class="pl-k">.+</span> bh)</td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">push!</span>(ys, hs[L])</td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span>           <span class="pl-c"><span class="pl-c">#</span> LSTM</span></td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@assert</span> r<span class="pl-k">.</span>inputMode <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">||</span> <span class="pl-c1">all</span>(w[<span class="pl-c1">1</span><span class="pl-k">:</span><span class="pl-c1">4</span><span class="pl-k">*</span>(<span class="pl-c1">1</span><span class="pl-k">+</span>r<span class="pl-k">.</span>direction)] <span class="pl-k">.==</span> <span class="pl-c1">nothing</span>)</td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> it = σ(Wixt + Riht-1 + bWi + bRi) </span></td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ft = σ(Wfxt + Rfht-1 + bWf + bRf) </span></td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ot = σ(Woxt + Roht-1 + bWo + bRo) </span></td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> c&#39;t = tanh(Wcxt + Rcht-1 + bWc + bRc) </span></td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ct = ft◦ct-1 + it◦c&#39;t </span></td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ht = ot◦tanh(ct)</span></td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">        c <span class="pl-k">=</span> cx<span class="pl-k">==</span><span class="pl-c1">nothing</span> ? <span class="pl-c1">fill!</span>(<span class="pl-c1">similar</span>(x,hsize),<span class="pl-c1">0</span>) <span class="pl-k">:</span> cx</td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">        cs <span class="pl-k">=</span> [ c[<span class="pl-k">:</span>,<span class="pl-k">:</span>,l] <span class="pl-k">for</span> l<span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">:</span>L ]</td>
      </tr>
      <tr>
        <td id="L678" class="blob-num js-line-number" data-line-number="678"></td>
        <td id="LC678" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> t <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>T</td>
      </tr>
      <tr>
        <td id="L679" class="blob-num js-line-number" data-line-number="679"></td>
        <td id="LC679" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> l <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>L</td>
      </tr>
      <tr>
        <td id="L680" class="blob-num js-line-number" data-line-number="680"></td>
        <td id="LC680" class="blob-code blob-code-inner js-file-line">                Wi,Wf,Wc,Wo,Ri,Rf,Rc,Ro <span class="pl-k">=</span> w[<span class="pl-c1">1</span><span class="pl-k">+</span><span class="pl-c1">8</span><span class="pl-k">*</span>(l<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">:</span><span class="pl-c1">8</span>l]</td>
      </tr>
      <tr>
        <td id="L681" class="blob-num js-line-number" data-line-number="681"></td>
        <td id="LC681" class="blob-code blob-code-inner js-file-line">                bWi,bWf,bWc,bWo,bRi,bRf,bRc,bRo <span class="pl-k">=</span> w[<span class="pl-c1">8</span>L<span class="pl-k">+</span><span class="pl-c1">1</span><span class="pl-k">+</span><span class="pl-c1">8</span><span class="pl-k">*</span>(l<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">:</span><span class="pl-c1">8</span>L<span class="pl-k">+</span><span class="pl-c1">8</span>l]</td>
      </tr>
      <tr>
        <td id="L682" class="blob-num js-line-number" data-line-number="682"></td>
        <td id="LC682" class="blob-code blob-code-inner js-file-line">                Wixt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wi<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wi<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L683" class="blob-num js-line-number" data-line-number="683"></td>
        <td id="LC683" class="blob-code blob-code-inner js-file-line">                Wfxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wf<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wf<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L684" class="blob-num js-line-number" data-line-number="684"></td>
        <td id="LC684" class="blob-code blob-code-inner js-file-line">                Wcxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wc<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wc<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L685" class="blob-num js-line-number" data-line-number="685"></td>
        <td id="LC685" class="blob-code blob-code-inner js-file-line">                Woxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wo<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wo<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L686" class="blob-num js-line-number" data-line-number="686"></td>
        <td id="LC686" class="blob-code blob-code-inner js-file-line">                it <span class="pl-k">=</span> <span class="pl-c1">sigm</span>.(Wixt <span class="pl-k">.+</span> Ri<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWi <span class="pl-k">.+</span> bRi)</td>
      </tr>
      <tr>
        <td id="L687" class="blob-num js-line-number" data-line-number="687"></td>
        <td id="LC687" class="blob-code blob-code-inner js-file-line">                ft <span class="pl-k">=</span> <span class="pl-c1">sigm</span>.(Wfxt <span class="pl-k">.+</span> Rf<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWf <span class="pl-k">.+</span> bRf)</td>
      </tr>
      <tr>
        <td id="L688" class="blob-num js-line-number" data-line-number="688"></td>
        <td id="LC688" class="blob-code blob-code-inner js-file-line">                ot <span class="pl-k">=</span> <span class="pl-c1">sigm</span>.(Woxt <span class="pl-k">.+</span> Ro<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWo <span class="pl-k">.+</span> bRo)</td>
      </tr>
      <tr>
        <td id="L689" class="blob-num js-line-number" data-line-number="689"></td>
        <td id="LC689" class="blob-code blob-code-inner js-file-line">                cn <span class="pl-k">=</span> <span class="pl-c1">tanh</span>.(Wcxt <span class="pl-k">.+</span> Rc<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWc <span class="pl-k">.+</span> bRc)</td>
      </tr>
      <tr>
        <td id="L690" class="blob-num js-line-number" data-line-number="690"></td>
        <td id="LC690" class="blob-code blob-code-inner js-file-line">                cs[l] <span class="pl-k">=</span> ft <span class="pl-k">.*</span> cs[l] <span class="pl-k">.+</span> it <span class="pl-k">.*</span> cn</td>
      </tr>
      <tr>
        <td id="L691" class="blob-num js-line-number" data-line-number="691"></td>
        <td id="LC691" class="blob-code blob-code-inner js-file-line">                hs[l] <span class="pl-k">=</span> ot <span class="pl-k">.*</span> <span class="pl-c1">tanh</span>.(cs[l])</td>
      </tr>
      <tr>
        <td id="L692" class="blob-num js-line-number" data-line-number="692"></td>
        <td id="LC692" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L693" class="blob-num js-line-number" data-line-number="693"></td>
        <td id="LC693" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">push!</span>(ys, hs[L])</td>
      </tr>
      <tr>
        <td id="L694" class="blob-num js-line-number" data-line-number="694"></td>
        <td id="LC694" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L695" class="blob-num js-line-number" data-line-number="695"></td>
        <td id="LC695" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">elseif</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">3</span>           <span class="pl-c"><span class="pl-c">#</span> GRU</span></td>
      </tr>
      <tr>
        <td id="L696" class="blob-num js-line-number" data-line-number="696"></td>
        <td id="LC696" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">@assert</span> r<span class="pl-k">.</span>inputMode <span class="pl-k">==</span> <span class="pl-c1">0</span> <span class="pl-k">||</span> <span class="pl-c1">all</span>(w[<span class="pl-c1">1</span><span class="pl-k">:</span><span class="pl-c1">3</span><span class="pl-k">*</span>(<span class="pl-c1">1</span><span class="pl-k">+</span>r<span class="pl-k">.</span>direction)] <span class="pl-k">.==</span> <span class="pl-c1">nothing</span>)</td>
      </tr>
      <tr>
        <td id="L697" class="blob-num js-line-number" data-line-number="697"></td>
        <td id="LC697" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> rt = σ(Wrxt + Rrht-1 + bWr + bRr)</span></td>
      </tr>
      <tr>
        <td id="L698" class="blob-num js-line-number" data-line-number="698"></td>
        <td id="LC698" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> it = σ(Wixt + Riht-1 + bWi + bRu)</span></td>
      </tr>
      <tr>
        <td id="L699" class="blob-num js-line-number" data-line-number="699"></td>
        <td id="LC699" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> h&#39;t = tanh(Whxt + rt◦(Rhht-1 + bRh) + bWh)</span></td>
      </tr>
      <tr>
        <td id="L700" class="blob-num js-line-number" data-line-number="700"></td>
        <td id="LC700" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> ht = (1 - it)◦h&#39;t + it◦ht-1</span></td>
      </tr>
      <tr>
        <td id="L701" class="blob-num js-line-number" data-line-number="701"></td>
        <td id="LC701" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> t <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>T</td>
      </tr>
      <tr>
        <td id="L702" class="blob-num js-line-number" data-line-number="702"></td>
        <td id="LC702" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> l <span class="pl-k">=</span> <span class="pl-c1">1</span><span class="pl-k">:</span>L</td>
      </tr>
      <tr>
        <td id="L703" class="blob-num js-line-number" data-line-number="703"></td>
        <td id="LC703" class="blob-code blob-code-inner js-file-line">                Wr,Wi,Wh,Rr,Ri,Rh <span class="pl-k">=</span> w[<span class="pl-c1">1</span><span class="pl-k">+</span><span class="pl-c1">6</span><span class="pl-k">*</span>(l<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">:</span><span class="pl-c1">6</span>l]</td>
      </tr>
      <tr>
        <td id="L704" class="blob-num js-line-number" data-line-number="704"></td>
        <td id="LC704" class="blob-code blob-code-inner js-file-line">                bWr,bWi,bWh,bRr,bRi,bRh <span class="pl-k">=</span> w[<span class="pl-c1">6</span>L<span class="pl-k">+</span><span class="pl-c1">1</span><span class="pl-k">+</span><span class="pl-c1">6</span><span class="pl-k">*</span>(l<span class="pl-k">-</span><span class="pl-c1">1</span>)<span class="pl-k">:</span><span class="pl-c1">6</span>L<span class="pl-k">+</span><span class="pl-c1">6</span>l]</td>
      </tr>
      <tr>
        <td id="L705" class="blob-num js-line-number" data-line-number="705"></td>
        <td id="LC705" class="blob-code blob-code-inner js-file-line">                Wrxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wr<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wr<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L706" class="blob-num js-line-number" data-line-number="706"></td>
        <td id="LC706" class="blob-code blob-code-inner js-file-line">                Wixt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wi<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wi<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L707" class="blob-num js-line-number" data-line-number="707"></td>
        <td id="LC707" class="blob-code blob-code-inner js-file-line">                Whxt <span class="pl-k">=</span> (l <span class="pl-k">&gt;</span> <span class="pl-c1">1</span> ? Wh<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l<span class="pl-k">-</span><span class="pl-c1">1</span>] <span class="pl-k">:</span> r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">0</span> ? Wh<span class="pl-k">&#39;</span> <span class="pl-k">*</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t] <span class="pl-k">:</span> x[<span class="pl-k">:</span>,<span class="pl-k">:</span>,t])</td>
      </tr>
      <tr>
        <td id="L708" class="blob-num js-line-number" data-line-number="708"></td>
        <td id="LC708" class="blob-code blob-code-inner js-file-line">                rt <span class="pl-k">=</span> <span class="pl-c1">sigm</span>.(Wrxt <span class="pl-k">.+</span> Rr<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWr <span class="pl-k">.+</span> bRr)</td>
      </tr>
      <tr>
        <td id="L709" class="blob-num js-line-number" data-line-number="709"></td>
        <td id="LC709" class="blob-code blob-code-inner js-file-line">                it <span class="pl-k">=</span> <span class="pl-c1">sigm</span>.(Wixt <span class="pl-k">.+</span> Ri<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bWi <span class="pl-k">.+</span> bRi)</td>
      </tr>
      <tr>
        <td id="L710" class="blob-num js-line-number" data-line-number="710"></td>
        <td id="LC710" class="blob-code blob-code-inner js-file-line">                ht <span class="pl-k">=</span> <span class="pl-c1">tanh</span>.(Whxt <span class="pl-k">.+</span> rt <span class="pl-k">.*</span> (Rh<span class="pl-k">&#39;</span> <span class="pl-k">*</span> hs[l] <span class="pl-k">.+</span> bRh) <span class="pl-k">.+</span> bWh)</td>
      </tr>
      <tr>
        <td id="L711" class="blob-num js-line-number" data-line-number="711"></td>
        <td id="LC711" class="blob-code blob-code-inner js-file-line">                hs[l] <span class="pl-k">=</span> (<span class="pl-c1">1</span> <span class="pl-k">.-</span> it) <span class="pl-k">.*</span> ht <span class="pl-k">.+</span> it <span class="pl-k">.*</span> hs[l]</td>
      </tr>
      <tr>
        <td id="L712" class="blob-num js-line-number" data-line-number="712"></td>
        <td id="LC712" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L713" class="blob-num js-line-number" data-line-number="713"></td>
        <td id="LC713" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">push!</span>(ys, hs[L])</td>
      </tr>
      <tr>
        <td id="L714" class="blob-num js-line-number" data-line-number="714"></td>
        <td id="LC714" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L715" class="blob-num js-line-number" data-line-number="715"></td>
        <td id="LC715" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">else</span></td>
      </tr>
      <tr>
        <td id="L716" class="blob-num js-line-number" data-line-number="716"></td>
        <td id="LC716" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">error</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>RNN not supported<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L717" class="blob-num js-line-number" data-line-number="717"></td>
        <td id="LC717" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L718" class="blob-num js-line-number" data-line-number="718"></td>
        <td id="LC718" class="blob-code blob-code-inner js-file-line">    y <span class="pl-k">=</span> <span class="pl-c1">reshape</span>(<span class="pl-c1">hcat</span>(ys<span class="pl-k">...</span>), ysize)</td>
      </tr>
      <tr>
        <td id="L719" class="blob-num js-line-number" data-line-number="719"></td>
        <td id="LC719" class="blob-code blob-code-inner js-file-line">    hyout <span class="pl-k">=</span> hy ? <span class="pl-c1">reshape</span>(<span class="pl-c1">hcat</span>(hs<span class="pl-k">...</span>), hsize) <span class="pl-k">:</span> <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L720" class="blob-num js-line-number" data-line-number="720"></td>
        <td id="LC720" class="blob-code blob-code-inner js-file-line">    cyout <span class="pl-k">=</span> cy <span class="pl-k">&amp;&amp;</span> r<span class="pl-k">.</span>mode <span class="pl-k">==</span> <span class="pl-c1">2</span> ? <span class="pl-c1">reshape</span>(<span class="pl-c1">hcat</span>(cs<span class="pl-k">...</span>), hsize) <span class="pl-k">:</span> <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L721" class="blob-num js-line-number" data-line-number="721"></td>
        <td id="LC721" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> (y,hyout,cyout,<span class="pl-c1">nothing</span>)</td>
      </tr>
      <tr>
        <td id="L722" class="blob-num js-line-number" data-line-number="722"></td>
        <td id="LC722" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L723" class="blob-num js-line-number" data-line-number="723"></td>
        <td id="LC723" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L724" class="blob-num js-line-number" data-line-number="724"></td>
        <td id="LC724" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> Hack for JLD file load/save of RNNs:</span></td>
      </tr>
      <tr>
        <td id="L725" class="blob-num js-line-number" data-line-number="725"></td>
        <td id="LC725" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> Pkg<span class="pl-k">.</span><span class="pl-c1">installed</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>JLD<span class="pl-pds">&quot;</span></span>) <span class="pl-k">!=</span> <span class="pl-c1">nothing</span></td>
      </tr>
      <tr>
        <td id="L726" class="blob-num js-line-number" data-line-number="726"></td>
        <td id="LC726" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">import</span> JLD<span class="pl-k">:</span> writeas, readas</td>
      </tr>
      <tr>
        <td id="L727" class="blob-num js-line-number" data-line-number="727"></td>
        <td id="LC727" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">type</span> RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; <span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L728" class="blob-num js-line-number" data-line-number="728"></td>
        <td id="LC728" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">writeas</span>(r<span class="pl-k">::</span><span class="pl-c1">RNN</span>) <span class="pl-k">=</span> <span class="pl-c1">RNNJLD</span>(r<span class="pl-k">.</span>inputSize, r<span class="pl-k">.</span>hiddenSize, r<span class="pl-k">.</span>numLayers, r<span class="pl-k">.</span>dropout, r<span class="pl-k">.</span>inputMode, r<span class="pl-k">.</span>direction, r<span class="pl-k">.</span>mode, r<span class="pl-k">.</span>algo, r<span class="pl-k">.</span>dataType)</td>
      </tr>
      <tr>
        <td id="L729" class="blob-num js-line-number" data-line-number="729"></td>
        <td id="LC729" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">readas</span>(r<span class="pl-k">::</span><span class="pl-c1">RNNJLD</span>) <span class="pl-k">=</span> <span class="pl-c1">rnninit</span>(r<span class="pl-k">.</span>inputSize, r<span class="pl-k">.</span>hiddenSize, numLayers<span class="pl-k">=</span>r<span class="pl-k">.</span>numLayers, dropout<span class="pl-k">=</span>r<span class="pl-k">.</span>dropout, skipInput<span class="pl-k">=</span>(r<span class="pl-k">.</span>inputMode<span class="pl-k">==</span><span class="pl-c1">1</span>), bidirectional<span class="pl-k">=</span>(r<span class="pl-k">.</span>direction<span class="pl-k">==</span><span class="pl-c1">1</span>), rnnType<span class="pl-k">=</span>(<span class="pl-k">:</span><span class="pl-c1">relu</span>,<span class="pl-k">:</span><span class="pl-c1">tanh</span>,<span class="pl-k">:</span><span class="pl-c1">lstm</span>,<span class="pl-k">:</span><span class="pl-c1">gru</span>)[<span class="pl-c1">1</span><span class="pl-k">+</span>r<span class="pl-k">.</span>mode], algo<span class="pl-k">=</span>r<span class="pl-k">.</span>algo, dataType<span class="pl-k">=</span>r<span class="pl-k">.</span>dataType)[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L730" class="blob-num js-line-number" data-line-number="730"></td>
        <td id="LC730" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L731" class="blob-num js-line-number" data-line-number="731"></td>
        <td id="LC731" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L732" class="blob-num js-line-number" data-line-number="732"></td>
        <td id="LC732" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> We need x[:,:,t] and hx[:,:,l]</span></td>
      </tr>
      <tr>
        <td id="L733" class="blob-num js-line-number" data-line-number="733"></td>
        <td id="LC733" class="blob-code blob-code-inner js-file-line"><span class="pl-k">using</span> Knet<span class="pl-k">:</span> Index3</td>
      </tr>
      <tr>
        <td id="L734" class="blob-num js-line-number" data-line-number="734"></td>
        <td id="LC734" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> Base<span class="pl-k">:</span> getindex, setindex!</td>
      </tr>
      <tr>
        <td id="L735" class="blob-num js-line-number" data-line-number="735"></td>
        <td id="LC735" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">getindex</span>(A<span class="pl-k">::</span><span class="pl-c1">KnetArray</span>, <span class="pl-k">::</span><span class="pl-c1">Colon</span>, <span class="pl-k">::</span><span class="pl-c1">Colon</span>, I<span class="pl-k">::</span><span class="pl-c1">Index3</span>)</td>
      </tr>
      <tr>
        <td id="L736" class="blob-num js-line-number" data-line-number="736"></td>
        <td id="LC736" class="blob-code blob-code-inner js-file-line">    B <span class="pl-k">=</span> <span class="pl-c1">reshape</span>(A, <span class="pl-c1">stride</span>(A,<span class="pl-c1">3</span>), <span class="pl-c1">size</span>(A,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L737" class="blob-num js-line-number" data-line-number="737"></td>
        <td id="LC737" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">reshape</span>(B[<span class="pl-k">:</span>,I], <span class="pl-c1">size</span>(A,<span class="pl-c1">1</span>), <span class="pl-c1">size</span>(A,<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L738" class="blob-num js-line-number" data-line-number="738"></td>
        <td id="LC738" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L739" class="blob-num js-line-number" data-line-number="739"></td>
        <td id="LC739" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">setindex!</span>(x<span class="pl-k">::</span><span class="pl-c1">KnetArray</span>, y, <span class="pl-k">::</span><span class="pl-c1">Colon</span>, <span class="pl-k">::</span><span class="pl-c1">Colon</span>, I<span class="pl-k">::</span><span class="pl-c1">Index3</span>)</td>
      </tr>
      <tr>
        <td id="L740" class="blob-num js-line-number" data-line-number="740"></td>
        <td id="LC740" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">reshape</span>(x, <span class="pl-c1">stride</span>(x,<span class="pl-c1">3</span>), <span class="pl-c1">size</span>(x,<span class="pl-c1">3</span>))[<span class="pl-k">:</span>,I] <span class="pl-k">=</span> y</td>
      </tr>
      <tr>
        <td id="L741" class="blob-num js-line-number" data-line-number="741"></td>
        <td id="LC741" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">return</span> x</td>
      </tr>
      <tr>
        <td id="L742" class="blob-num js-line-number" data-line-number="742"></td>
        <td id="LC742" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L743" class="blob-num js-line-number" data-line-number="743"></td>
        <td id="LC743" class="blob-code blob-code-inner js-file-line"><span class="pl-k">function</span> <span class="pl-en">getindex</span><span class="pl-c1">{T,I&lt;:Integer}</span>(x<span class="pl-k">::</span><span class="pl-c1">KnetArray{T,2}</span>, <span class="pl-k">::</span><span class="pl-c1">Colon</span>, m<span class="pl-k">::</span><span class="pl-c1">Array{I,2}</span>)</td>
      </tr>
      <tr>
        <td id="L744" class="blob-num js-line-number" data-line-number="744"></td>
        <td id="LC744" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">reshape</span>(x[<span class="pl-k">:</span>,<span class="pl-c1">vec</span>(m)], <span class="pl-c1">size</span>(x,<span class="pl-c1">1</span>), <span class="pl-c1">size</span>(m,<span class="pl-c1">1</span>), <span class="pl-c1">size</span>(m,<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L745" class="blob-num js-line-number" data-line-number="745"></td>
        <td id="LC745" class="blob-code blob-code-inner js-file-line"><span class="pl-k">end</span></td>
      </tr>
      <tr>
        <td id="L746" class="blob-num js-line-number" data-line-number="746"></td>
        <td id="LC746" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
</table>

  <div class="BlobToolbar position-absolute js-file-line-actions dropdown js-menu-container js-select-menu d-none" aria-hidden="true">
    <button class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1 dropdown-toggle js-menu-target" id="js-file-line-action-button" type="button" aria-expanded="false" aria-haspopup="true" aria-label="Inline file action toolbar" aria-controls="inline-file-actions">
      <svg aria-hidden="true" class="octicon octicon-kebab-horizontal" height="16" version="1.1" viewBox="0 0 13 16" width="13"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/></svg>
    </button>
    <div class="dropdown-menu-content js-menu-content" id="inline-file-actions">
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2">
        <li><a class="js-zeroclipboard dropdown-item" style="cursor:pointer;" id="js-copy-lines" data-original-text="Copy lines">Copy lines</a></li>
        <li><a class="js-zeroclipboard dropdown-item" id= "js-copy-permalink" style="cursor:pointer;" data-original-text="Copy permalink">Copy permalink</a></li>
        <li><a href="/denizyuret/Knet.jl/blame/56d3fbc2799ddb022df688418639da6b85be9a35/src/rnn.jl" class="dropdown-item js-update-url-with-hash" id="js-view-git-blame">View git blame</a></li>
          <li><a href="/denizyuret/Knet.jl/issues/new" class="dropdown-item" id="js-new-issue">Open new issue</a></li>
      </ul>
    </div>
  </div>

  </div>

  </div>

  <button type="button" data-facebox="#jump-to-line" data-facebox-class="linejump" data-hotkey="l" class="d-none">Jump to Line</button>
  <div id="jump-to-line" style="display:none">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
      <input class="form-control linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
      <button type="submit" class="btn">Go</button>
</form>  </div>

  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

      
<div class="footer container-lg px-3" role="contentinfo">
  <div class="position-relative d-flex flex-justify-between py-6 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap ">
      <li class="mr-3">&copy; 2017 <span title="0.14203s from unicorn-3014522701-5kh3f">GitHub</span>, Inc.</li>
        <li class="mr-3"><a href="https://github.com/site/terms" data-ga-click="Footer, go to terms, text:terms">Terms</a></li>
        <li class="mr-3"><a href="https://github.com/site/privacy" data-ga-click="Footer, go to privacy, text:privacy">Privacy</a></li>
        <li class="mr-3"><a href="https://github.com/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li class="mr-3"><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a href="https://help.github.com" data-ga-click="Footer, go to help, text:help">Help</a></li>
    </ul>

    <a href="https://github.com" aria-label="Homepage" class="footer-octicon" title="GitHub">
      <svg aria-hidden="true" class="octicon octicon-mark-github" height="24" version="1.1" viewBox="0 0 16 16" width="24"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
    <ul class="list-style-none d-flex flex-wrap ">
        <li class="mr-3"><a href="https://github.com/contact" data-ga-click="Footer, go to contact, text:contact">Contact GitHub</a></li>
      <li class="mr-3"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li class="mr-3"><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li class="mr-3"><a href="https://github.com/blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a href="https://github.com/about" data-ga-click="Footer, go to about, text:about">About</a></li>

    </ul>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
    You can't perform that action at this time.
  </div>


    <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/compat-e42a8bf9c380758734e39851db04de7cbeeb2f3860efbd481c96ac12c25a6ecb.js"></script>
    <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-f6afcaeac971138fed77bef35a12efac982cf10f6d34190912b7307946f14c8b.js"></script>
    
    <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-0bf824ad9b5c44a4ba98c8d86ede76a76d1bf24675f586cb61210f717b420cbe.js"></script>
    
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
  </div>
</div>


  </body>
</html>

